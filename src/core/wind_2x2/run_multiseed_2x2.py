"""Run the 9 paired training seeds excluding seed 42 for all four Grade-TSMixer Wind 2x2 cells.

Seed 42 is deliberately excluded: its formal existing artifacts are reused by
the separate summarizer. Completed runs are validated and skipped on resume.
"""
from __future__ import annotations

import hashlib, importlib.metadata, json, pickle, platform, sys, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from evaluate.metrics import TARGET_NAMES, compute_all_metrics
from models.top_ts_models import build_model

PROCESSED = ROOT / "data" / "processed"
STANDARDIZED = PROCESSED / "standardized"
OUT = HERE / "multiseed"
SEEDS = [123, 456, 789, 2024, 3407, 5678, 7777, 8888, 9999]
CONFIGS = {
    "O5": (HERE / "data" / "X_original5_scaled.npy", 5),
    "O6": (STANDARDIZED / "X_original6_scaled.npy", 6),
    "W6": (STANDARDIZED / "X_wind6_scaled.npy", 6),
    "W7": (STANDARDIZED / "X_wind7_scaled.npy", 7),
}
MAX_EPOCHS, BATCH_SIZE, PATIENCE = 300, 32, 30
LR, WEIGHT_DECAY = 1e-3, 1e-4

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def set_seed(seed: int) -> None:
    torch.set_num_threads(8)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inverse_y(x: np.ndarray, scaler: dict) -> np.ndarray:
    return x * scaler["std"] + scaler["mean"]

def valid_complete(tables: Path, tag: str) -> bool:
    needed = [tables / f"{tag}_metrics.csv", tables / f"{tag}_predictions_wide.csv",
              tables / f"{tag}_training_history.csv", tables / f"{tag}_run_metadata.json"]
    if not all(p.exists() and p.stat().st_size > 0 for p in needed): return False
    try:
        return len(pd.read_csv(needed[1])) == 420 and len(pd.read_csv(needed[0])) == 4
    except Exception: return False

def run_one(seed: int, config: str, common: dict) -> None:
    run_dir = OUT / f"seed{seed}" / config.lower()
    tables, checkpoints, logs = run_dir / "tables", run_dir / "checkpoints", run_dir / "logs"
    for p in (tables, checkpoints, logs): p.mkdir(parents=True, exist_ok=True)
    tag = f"grade_tsmixer_{config.lower()}_default_seed{seed}"
    if valid_complete(tables, tag):
        print(f"SKIP complete {tag}", flush=True); return
    log_path = logs / f"{tag}.log"
    def log(msg: str) -> None:
        line = f"{datetime.now().astimezone().isoformat()} {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f: f.write(line + "\n")
    log_path.write_text("", encoding="utf-8")
    started = datetime.now().astimezone(); tic = time.perf_counter()
    set_seed(seed)
    x_path, input_dim = CONFIGS[config]
    X = np.load(x_path)
    if X.shape != (2790, 153, input_dim): raise ValueError(f"Bad {config} shape {X.shape}")
    tr, va, te = common["train"], common["val"], common["test"]
    y, grades = common["y_scaled"], common["grades"]
    train_ds = TensorDataset(torch.as_tensor(X[tr], dtype=torch.float32),
                             torch.as_tensor(y[tr], dtype=torch.float32),
                             torch.as_tensor(grades[tr], dtype=torch.long))
    val_ds = TensorDataset(torch.as_tensor(X[va], dtype=torch.float32),
                           torch.as_tensor(y[va], dtype=torch.float32),
                           torch.as_tensor(grades[va], dtype=torch.long))
    test_ds = TensorDataset(torch.as_tensor(X[te], dtype=torch.float32),
                            torch.as_tensor(grades[te], dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("grade_tsmixer", input_dim=input_dim, seq_len=153,
                        variant="default", num_grades=16).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss(); best_loss = float("inf"); best_state = None
    best_epoch = 0; wait = 0; history = []
    log(f"START {tag} device={device} params={params} X={X.shape}")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); total = 0.0
        for bx, by, bg in train_loader:
            bx, by, bg = bx.to(device), by.to(device), bg.to(device)
            optimizer.zero_grad(); loss = criterion(model(bx, bg), by)
            loss.backward(); optimizer.step(); total += loss.item() * len(bx)
        train_loss = total / len(train_ds)
        model.eval(); total = 0.0
        with torch.no_grad():
            for bx, by, bg in val_loader:
                bx, by, bg = bx.to(device), by.to(device), bg.to(device)
                total += criterion(model(bx, bg), by).item() * len(bx)
        val_loss = total / len(val_ds); scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if val_loss < best_loss:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: wait += 1
        if epoch == 1 or epoch % 10 == 0: log(f"epoch={epoch} train={train_loss:.8f} val={val_loss:.8f}")
        if wait >= PATIENCE: break
    stop_epoch = epoch
    if best_state is None: raise RuntimeError("No best state")
    model.load_state_dict(best_state)
    pred_scaled = []
    model.eval()
    with torch.no_grad():
        for bx, bg in test_loader: pred_scaled.append(model(bx.to(device), bg.to(device)).cpu().numpy())
    y_true = inverse_y(y[te], common["y_scaler"])
    y_pred = inverse_y(np.concatenate(pred_scaled), common["y_scaler"])
    metrics = compute_all_metrics(y_true, y_pred)
    history_df = pd.DataFrame(history)
    history_df.to_csv(tables / f"{tag}_training_history.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(tables / f"{tag}_metrics.csv", index=False, encoding="utf-8-sig")
    metrics.to_json(tables / f"{tag}_metrics.json", orient="records", indent=2, force_ascii=False)
    wide = []
    for si, oi in enumerate(te):
        row = {"sample_index": si, "original_index": int(oi), "grade_id": int(grades[oi]), "seed": seed}
        for j, target in enumerate(TARGET_NAMES):
            row[f"true_{target}"] = float(y_true[si,j]); row[f"pred_{target}"] = float(y_pred[si,j])
        wide.append(row)
    pd.DataFrame(wide).to_csv(tables / f"{tag}_predictions_wide.csv", index=False, encoding="utf-8-sig")
    torch.save({"model_state_dict": best_state, "config": config, "input_dim": input_dim,
                "seed": seed, "best_epoch": best_epoch, "stop_epoch": stop_epoch,
                "best_val_loss": best_loss, "target_names": TARGET_NAMES},
               checkpoints / f"{tag}_best.pth")
    finished = datetime.now().astimezone()
    metadata = {"tag": tag, "seed": seed, "configuration": config, "input_dim": input_dim,
                "features_file": str(x_path), "features_sha256": sha256(x_path),
                "fixed_split_sha256": common["split_hash"], "split_sizes": [1952,418,420],
                "parameter_count": params, "best_epoch": best_epoch, "stop_epoch": stop_epoch,
                "best_validation_mse": best_loss, "duration_seconds": time.perf_counter()-tic,
                "started_at": started.isoformat(), "finished_at": finished.isoformat(),
                "training": {"optimizer":"AdamW","lr":LR,"weight_decay":WEIGHT_DECAY,
                             "batch_size":BATCH_SIZE,"max_epochs":MAX_EPOCHS,
                             "scheduler":"ReduceLROnPlateau(factor=0.5,patience=10)",
                             "early_stopping_patience":PATIENCE,"loss":"MSE"},
                "environment": {"python":platform.python_version(),"executable":sys.executable,
                                "torch":torch.__version__,"numpy":np.__version__,
                                "pandas":pd.__version__,"device":str(device)}}
    (tables / f"{tag}_run_metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    avg_mape=float(metrics.loc[metrics.metric=="MAPE","Avg"].iloc[0])
    log(f"DONE {tag} best={best_epoch} stop={stop_epoch} avg_mape={avg_mape:.8f} seconds={metadata['duration_seconds']:.2f}")

def main() -> None:
    split_path = PROCESSED / "split_indices.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    common = {"train":np.asarray(split["train_idx"]),"val":np.asarray(split["val_idx"]),
              "test":np.asarray(split["test_idx"]),"split_hash":sha256(split_path),
              "y_scaled":np.load(STANDARDIZED / "y_scaled.npy"),
              "grades":np.load(PROCESSED / "grade_ids.npy")}
    with (PROCESSED / "scalers" / "y_scaler.pkl").open("rb") as f: common["y_scaler"] = pickle.load(f)
    if tuple(map(len,(common["train"],common["val"],common["test"]))) != (1952,418,420):
        raise ValueError("Split changed")
    failures=[]
    for seed in SEEDS:
        for config in ("O5","O6","W6","W7"):
            try: run_one(seed,config,common)
            except Exception as exc:
                failures.append({"seed":seed,"config":config,"error":repr(exc),"traceback":traceback.format_exc()})
                print(f"FAILED seed={seed} config={config}: {exc}",flush=True)
    (OUT / "run_failures.json").write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding="utf-8")
    if failures: raise RuntimeError(f"{len(failures)} runs failed; see run_failures.json")
    print("ALL 36 RUNS COMPLETE",flush=True)

if __name__ == "__main__": main()
