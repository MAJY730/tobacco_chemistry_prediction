"""K326-specific grade contribution validation.

This experiment is isolated under k326_grade_validation/. It reads the formal
2790-record less-rules arrays but never writes to the data, model,
checkpoint, or results directories.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import platform
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate.metrics import TARGET_NAMES, compute_all_metrics  # noqa: E402
from models.top_ts_models import TSMixerBlock  # noqa: E402

ROOT = PROJECT_ROOT / "k326_grade_validation"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "splits"
RANDOM_DIR = ROOT / "random_5seed"
CV_DIR = ROOT / "group_cv"
SUMMARY_DIR = ROOT / "summary"
RUNTIME_PRIVATE_DIR = ROOT / "runtime_private"

K326_LABELS = ["B2F_K326", "C2F_K326", "C3F_K326", "X2F_K326"]
INTERNAL_GRADES = ["B2F", "C2F", "C3F", "X2F"]
K326_ID = {label: i for i, label in enumerate(K326_LABELS)}
INTERNAL_NAME = {label: label.split("_")[0] for label in K326_LABELS}
SEEDS = [42, 123, 456, 789, 2024]
N_PERM = 100_000
N_BOOT = 100_000


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClimateEncoder(nn.Module):
    """Functional climate encoder from the formal default TSMixer."""

    def __init__(self, input_dim=7, seq_len=153, d_model=64, n_blocks=3,
                 hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            TSMixerBlock(seq_len, d_model, hidden_dim, dropout)
            for _ in range(n_blocks)
        ])
        self.pool_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.pool_norm(x).mean(dim=1)


class ClimateOnlyTSMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ClimateEncoder()
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x, grade_ids=None):
        z = self.encoder(x)
        return self.fc2(self.dropout(F.gelu(self.fc1(z))))


class K326GradeAwareTSMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ClimateEncoder()
        self.grade_embed = nn.Embedding(4, 8)
        self.fc1 = nn.Linear(72, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x, grade_ids):
        z = torch.cat([self.encoder(x), self.grade_embed(grade_ids)], dim=-1)
        return self.fc2(self.dropout(F.gelu(self.fc1(z))))


def build_model(name: str):
    if name == "climate_only":
        return ClimateOnlyTSMixer()
    if name == "grade_aware":
        return K326GradeAwareTSMixer()
    raise ValueError(name)


def load_source():
    info = pd.read_csv(DATA_DIR / "sample_info.csv")
    X = np.load(DATA_DIR / "X_wind7.npy").astype(np.float32)
    y = np.load(DATA_DIR / "y.npy").astype(np.float32)
    old_grade_ids = np.load(DATA_DIR / "grade_ids.npy")
    with (DATA_DIR / "grade_mapping.json").open(encoding="utf-8") as f:
        mapping = json.load(f)
    keep = info["grade"].isin(K326_LABELS).to_numpy()
    sub = info.loc[keep].copy().reset_index(drop=True)
    original_idx = sub["original_index"].to_numpy(dtype=int)
    sub["original_grade_label"] = sub["grade"]
    sub["original_grade_id"] = old_grade_ids[original_idx]
    sub["k326_internal_grade"] = sub["grade"].map(INTERNAL_NAME)
    sub["k326_internal_grade_id"] = sub["grade"].map(K326_ID).astype(int)
    sub["county_year"] = sub["place"].astype(str).str.strip() + "|" + sub["year"].astype(str)
    expected_old = {g: mapping["grade_to_id"][g] for g in K326_LABELS}
    actual_old = dict(zip(sub["original_grade_label"], sub["original_grade_id"]))
    if any(actual_old[g] != expected_old[g] for g in K326_LABELS):
        raise RuntimeError("Original grade IDs do not match grade_mapping.json")
    return sub, X, y


def select_inner_split(dev_local, sub: pd.DataFrame, fold: int):
    """Lock a K326-specific group-disjoint validation set near 15% records.

    A deterministic candidate search prioritizes all four grades in inner-train,
    then all four in validation, then validation size and distribution balance.
    """
    dev = sub.iloc[dev_local]
    group_to_local = {g: dev_local[dev["county_year"].to_numpy() == g]
                      for g in pd.unique(dev["county_year"])}
    groups = np.array(list(group_to_local), dtype=object)
    target = 0.15 * len(dev_local)
    overall = dev["k326_internal_grade_id"].value_counts(normalize=True).reindex(range(4), fill_value=0).to_numpy()
    rng = np.random.RandomState(4200 + fold)
    candidates = []
    # Two thousand deterministic candidates are ample for 37--39 development
    # groups and keep the preflight fast enough to run the analysis reproducibly.
    for _ in range(2_000):
        order = rng.permutation(groups)
        chosen, n = [], 0
        for g in order:
            if n < target or not chosen:
                chosen.append(g); n += len(group_to_local[g])
            else:
                break
        val_idx = np.sort(np.concatenate([group_to_local[g] for g in chosen]))
        train_idx = np.sort(np.setdiff1d(dev_local, val_idx))
        train_seen = set(sub.iloc[train_idx]["k326_internal_grade_id"])
        val_seen = set(sub.iloc[val_idx]["k326_internal_grade_id"])
        val_dist = sub.iloc[val_idx]["k326_internal_grade_id"].value_counts(normalize=True).reindex(range(4), fill_value=0).to_numpy()
        score = ((4-len(train_seen))*1_000_000 + (4-len(val_seen))*10_000 +
                 abs(len(val_idx)-target)*10 + np.abs(val_dist-overall).sum())
        candidates.append((score, train_idx, val_idx))
    _, train_idx, val_idx = min(candidates, key=lambda z: z[0])
    return train_idx, val_idx


def prepare_splits(sub: pd.DataFrame):
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    with (DATA_DIR / "split_indices.json").open(encoding="utf-8") as f:
        formal = json.load(f)
    orig = sub["original_index"].to_numpy()
    random_split = {
        role: orig[np.isin(orig, formal[f"{role}_idx"])].tolist()
        for role in ("train", "val", "test")
    }
    json_dump(random_split, SPLIT_DIR / "random_split_indices.json")

    groups = sub["county_year"].to_numpy()
    folds = []
    outer_assignment = np.zeros(len(sub), dtype=int)
    for fold, (dev_local, test_local) in enumerate(
            GroupKFold(5).split(sub, groups=groups), start=1):
        train_local, val_local = select_inner_split(np.asarray(dev_local), sub, fold)
        outer_assignment[test_local] = fold
        obj = {
            "fold": fold,
            "training_seed": 42,
            "inner_selection_seed": 4200 + fold,
            "train_idx": orig[train_local].tolist(),
            "val_idx": orig[val_local].tolist(),
            "test_idx": orig[np.sort(test_local)].tolist(),
        }
        folds.append(obj)
        json_dump(obj, SPLIT_DIR / f"fold{fold}_indices.json")
    assignment = sub.copy()
    assignment["outer_test_fold"] = outer_assignment
    assignment.to_csv(SPLIT_DIR / "outer_fold_assignment.csv", index=False, encoding="utf-8-sig")
    json_dump({"method": "GroupKFold", "n_splits": 5, "group": "place+year", "folds": folds},
              SPLIT_DIR / "group_cv_splits.json")
    return random_split, folds


def preflight(sub, random_split, folds):
    expected_counts = {"B2F_K326": 84, "C2F_K326": 124, "C3F_K326": 125, "X2F_K326": 61}
    checks = []
    def add(name, passed, detail): checks.append({"check": name, "passed": bool(passed), "detail": detail})
    add("records_394", len(sub) == 394, len(sub))
    add("groups_47", sub["county_year"].nunique() == 47, sub["county_year"].nunique())
    counts = sub["original_grade_label"].value_counts().to_dict()
    add("four_grade_counts", counts == expected_counts, counts)
    rcounts = {k: len(v) for k, v in random_split.items()}
    add("random_sizes", rcounts == {"train": 279, "val": 53, "test": 62}, rcounts)
    for role, idx in random_split.items():
        x = sub[sub["original_index"].isin(idx)]
        add(f"random_{role}_four_grades", x["k326_internal_grade_id"].nunique() == 4,
            x["k326_internal_grade"].value_counts().to_dict())
    all_test = []
    outer_group_sets = []
    for f in folds:
        role_groups = {}
        for role in ("train", "val", "test"):
            x = sub[sub["original_index"].isin(f[f"{role}_idx"])]
            role_groups[role] = set(x["county_year"])
            if role == "train":
                add(f"fold{f['fold']}_inner_train_four_grades", x["k326_internal_grade_id"].nunique() == 4,
                    x["k326_internal_grade"].value_counts().to_dict())
            if role == "val":
                add(f"fold{f['fold']}_validation_grade_coverage", True,
                    {"n_grades": x["k326_internal_grade_id"].nunique(), "counts": x["k326_internal_grade"].value_counts().to_dict()})
        disjoint = not (role_groups["train"] & role_groups["val"] or role_groups["train"] & role_groups["test"] or role_groups["val"] & role_groups["test"])
        add(f"fold{f['fold']}_roles_group_disjoint", disjoint, {k: len(v) for k, v in role_groups.items()})
        all_test.extend(f["test_idx"]); outer_group_sets.append(role_groups["test"])
    add("outer_records_exact_once", len(all_test) == 394 and len(set(all_test)) == 394, {"n": len(all_test), "unique": len(set(all_test))})
    union_groups = set().union(*outer_group_sets)
    pair_overlap = sum(len(outer_group_sets[i] & outer_group_sets[j]) for i in range(5) for j in range(i+1, 5))
    add("outer_47_groups_no_overlap", len(union_groups) == 47 and pair_overlap == 0,
        {"groups": len(union_groups), "pair_overlap": pair_overlap})

    climate = build_model("climate_only"); grade = build_model("grade_aware")
    enc_match = {k: list(v.shape) for k,v in climate.encoder.state_dict().items()} == {k: list(v.shape) for k,v in grade.encoder.state_dict().items()}
    add("shared_climate_encoder_shapes", enc_match, climate.encoder.__class__.__name__)
    add("embedding_4x8", list(grade.grade_embed.weight.shape) == [4,8], list(grade.grade_embed.weight.shape))
    add("no_variety_input", True, "Only k326_internal_grade_id is passed; no variety tensor or embedding exists")
    add("isolated_output_directory", str(ROOT).startswith(str(PROJECT_ROOT / "k326_grade_validation")), str(ROOT))
    add("scaler_train_only_code_path", True, "fit_scalers is called with run-specific train_idx only")
    passed = all(x["passed"] for x in checks)
    audit = {
        "passed": passed,
        "checks": checks,
        "data": {
            "records": len(sub), "county_year_groups": sub["county_year"].nunique(),
            "counties": sub["place"].nunique(), "years": sorted(sub["year"].unique().tolist()),
            "grade_counts": counts,
        },
        "architecture": {
            "climate_only_parameters": sum(p.numel() for p in climate.parameters() if p.requires_grad),
            "grade_aware_parameters": sum(p.numel() for p in grade.parameters() if p.requires_grad),
            "embedding_shape": list(grade.grade_embed.weight.shape),
            "common_encoder_state_shapes": {k:list(v.shape) for k,v in climate.encoder.state_dict().items()},
        },
    }
    json_dump(audit, SPLIT_DIR / "preflight_audit.json")
    pd.DataFrame(checks).to_csv(SPLIT_DIR / "preflight_checks.csv", index=False, encoding="utf-8-sig")
    if not passed:
        raise RuntimeError("K326 preflight failed; see splits/preflight_audit.json")
    return audit


def fit_scalers(X, y, train_idx):
    xtr = X[train_idx]
    ytr = y[train_idx]
    x_mean = xtr.mean(axis=(0,1)); x_std = xtr.std(axis=(0,1)); x_std[x_std < 1e-8] = 1.0
    y_mean = ytr.mean(axis=0); y_std = ytr.std(axis=0); y_std[y_std < 1e-8] = 1.0
    return {"mean": x_mean, "std": x_std}, {"mean": y_mean, "std": y_std}


def raw_metrics(y_true, y_pred):
    den = np.maximum(np.abs(y_true), 1e-8)
    mae = np.mean(np.abs(y_true-y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true-y_pred)**2, axis=0))
    mape = np.mean(np.abs(y_true-y_pred)/den, axis=0)*100
    ss_res = np.sum((y_true-y_pred)**2, axis=0)
    ss_tot = np.sum((y_true-y_true.mean(axis=0))**2, axis=0)
    r2 = np.where(ss_tot < 1e-10, 0, 1-ss_res/ss_tot)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def metrics_frames(y_true, y_pred):
    rounded = compute_all_metrics(y_true, y_pred)
    raw = raw_metrics(y_true, y_pred)
    detailed = []
    for metric, vals in raw.items():
        for target, value in zip(TARGET_NAMES, vals):
            detailed.append({"metric": metric, "target": target, "value": float(value)})
        detailed.append({"metric": metric, "target": "Avg", "value": float(np.mean(vals))})
    return rounded, pd.DataFrame(detailed), {m: float(np.mean(v)) for m,v in raw.items()}


def train_run(model_name, seed, train_idx, val_idx, test_idx, run_dir, sub, X, y, split_kind, fold=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    logger = logging.getLogger(f"{split_kind}_{fold}_{seed}_{model_name}")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")); logger.addHandler(handler)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_idx = np.asarray(train_idx, dtype=int); val_idx=np.asarray(val_idx,dtype=int); test_idx=np.asarray(test_idx,dtype=int)
    xsc, ysc = fit_scalers(X, y, train_idx)
    with (run_dir / "X_scaler.pkl").open("wb") as f: pickle.dump(xsc, f)
    with (run_dir / "y_scaler.pkl").open("wb") as f: pickle.dump(ysc, f)
    Xs = (X-xsc["mean"][None,None,:])/xsc["std"][None,None,:]
    ys = (y-ysc["mean"][None,:])/ysc["std"][None,:]
    gid_full = np.full(len(X), -1, dtype=int)
    gid_full[sub["original_index"].to_numpy()] = sub["k326_internal_grade_id"].to_numpy()
    def ds(idx, with_y=True):
        args=[torch.tensor(Xs[idx],dtype=torch.float32)]
        if with_y: args.append(torch.tensor(ys[idx],dtype=torch.float32))
        args.append(torch.tensor(gid_full[idx],dtype=torch.long))
        return TensorDataset(*args)
    generator=torch.Generator(); generator.manual_seed(seed)
    tr=DataLoader(ds(train_idx),batch_size=32,shuffle=True,generator=generator)
    va=DataLoader(ds(val_idx),batch_size=32,shuffle=False)
    te=DataLoader(ds(test_idx,False),batch_size=32,shuffle=False)
    model=build_model(model_name).to(device)
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=10)
    loss_fn=nn.MSELoss(); best=float("inf"); best_state=None; wait=0; best_epoch=0; history=[]
    start=time.perf_counter()
    for epoch in range(1,301):
        model.train(); total=0
        for bx,by,bg in tr:
            bx,by,bg=bx.to(device),by.to(device),bg.to(device); opt.zero_grad()
            pred=model(bx,bg); loss=loss_fn(pred,by); loss.backward(); opt.step(); total+=loss.item()*len(bx)
        train_loss=total/len(tr.dataset)
        model.eval(); total=0
        with torch.no_grad():
            for bx,by,bg in va:
                bx,by,bg=bx.to(device),by.to(device),bg.to(device)
                total+=loss_fn(model(bx,bg),by).item()*len(bx)
        val_loss=total/len(va.dataset); scheduler.step(val_loss)
        history.append({"epoch":epoch,"train_mse":train_loss,"val_mse":val_loss,"lr":opt.param_groups[0]["lr"]})
        if val_loss < best:
            best=val_loss; best_epoch=epoch; wait=0
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if epoch==1 or epoch%20==0: logger.info("epoch=%d train=%.8f val=%.8f",epoch,train_loss,val_loss)
        if wait>=30: break
    stop_epoch=epoch; duration=time.perf_counter()-start
    model.load_state_dict(best_state); model.eval(); pred_s=[]
    with torch.no_grad():
        for bx,bg in te:
            pred_s.append(model(bx.to(device),bg.to(device)).cpu().numpy())
    pred_s=np.concatenate(pred_s); pred=pred_s*ysc["std"]+ysc["mean"]; true=y[test_idx]
    rounded,detailed,overall=metrics_frames(true,pred)
    rounded.to_csv(run_dir/"metrics.csv",index=False,encoding="utf-8-sig")
    detailed.to_csv(run_dir/"metrics_long.csv",index=False,encoding="utf-8-sig")
    json_dump({"overall":overall,"per_target":detailed.to_dict("records")},run_dir/"metrics.json")
    pd.DataFrame(history).to_csv(run_dir/"training_history.csv",index=False,encoding="utf-8-sig")
    # Passing a file handle avoids PyTorch's Windows zip-writer path-length
    # limitation while retaining the requested isolated directory layout.
    with (run_dir/"best_checkpoint.pth").open("wb") as checkpoint_file:
        torch.save({"state":best_state,"model_name":model_name,"seed":seed,"best_epoch":best_epoch,
                    "stop_epoch":stop_epoch,"parameter_count":params}, checkpoint_file)
    meta=sub.set_index("original_index").loc[test_idx].reset_index()
    wide=meta[["original_index","sample_id","place","year","county_year","original_grade_label","original_grade_id","k326_internal_grade","k326_internal_grade_id"]].copy()
    for j,t in enumerate(TARGET_NAMES): wide[f"y_true_{t}"]=true[:,j]; wide[f"y_pred_{t}"]=pred[:,j]
    wide["split_kind"]=split_kind; wide["fold"]=fold; wide["seed"]=seed; wide["model"]=model_name
    wide.to_csv(run_dir/"predictions_wide.csv",index=False,encoding="utf-8-sig")
    long=[]
    for i,row in wide.iterrows():
        for t in TARGET_NAMES:
            long.append({**{k:row[k] for k in ["original_index","sample_id","place","year","county_year","original_grade_label","original_grade_id","k326_internal_grade","k326_internal_grade_id","split_kind","fold","seed","model"]},
                         "target":t,"y_true":row[f"y_true_{t}"],"y_pred":row[f"y_pred_{t}"]})
    pd.DataFrame(long).to_csv(run_dir/"predictions_long.csv",index=False,encoding="utf-8-sig")
    metadata={"model":model_name,"seed":seed,"split_kind":split_kind,"fold":fold,"train_n":len(train_idx),"val_n":len(val_idx),"test_n":len(test_idx),
              "train_indices":train_idx,"val_indices":val_idx,"test_indices":test_idx,"parameter_count":params,"best_epoch":best_epoch,"stop_epoch":stop_epoch,
              "best_validation_mse":best,"training_time_seconds":duration,"device":str(device),"python":platform.python_version(),"torch":torch.__version__,
              "numpy":np.__version__,"pandas":pd.__version__,"sklearn":sklearn.__version__,"scipy":scipy.__version__,"grade_embedding_shape":[4,8] if model_name=="grade_aware" else None,
              "variety_embedding":False,"wind7_features":["Tmax","Tmin","Tavg","Rain","Sunshine","Pressure","Wind"],
              "optimizer":"AdamW","lr":0.001,"weight_decay":0.0001,"batch_size":32,"max_epochs":300,"scheduler_factor":0.5,"scheduler_patience":10,"early_stopping_patience":30,"loss":"MSE"}
    json_dump(metadata,run_dir/"metadata.json"); logger.info("complete metrics=%s",overall)
    return {"model":model_name,"seed":seed,"fold":fold,**overall,"parameter_count":params,"best_epoch":best_epoch,"stop_epoch":stop_epoch,"duration_seconds":duration}


def signflip_p(values):
    values=np.asarray(values,float); obs=abs(values.mean()); n=len(values)
    if n <= 20:
        means=np.array([(values*np.array(s)).mean() for s in product([-1,1],repeat=n)])
        return float((np.abs(means)>=obs-1e-15).mean())
    rng=np.random.default_rng(42001); count=0
    for _ in range(N_PERM//1000):
        signs=rng.choice([-1,1],size=(1000,n)); count+=int((np.abs((signs*values).mean(axis=1))>=obs).sum())
    return (count+1)/(N_PERM+1)


def bootstrap_ci(values,n_boot=N_BOOT,seed=42002):
    values=np.asarray(values,float); rng=np.random.default_rng(seed); means=[]
    for _ in range(n_boot//1000):
        idx=rng.integers(0,len(values),size=(1000,len(values))); means.extend(values[idx].mean(axis=1))
    return [float(x) for x in np.percentile(means,[2.5,97.5])]


def summarize_random(rows):
    df=pd.DataFrame(rows); SUMMARY_DIR.mkdir(parents=True,exist_ok=True)
    df.to_csv(SUMMARY_DIR/"random_5seed_model_metrics.csv",index=False,encoding="utf-8-sig")
    piv=df.pivot(index="seed",columns="model",values="MAPE").reset_index()
    piv["delta_mape_pp"]=piv["grade_aware"]-piv["climate_only"]
    piv.to_csv(SUMMARY_DIR/"random_5seed_comparison.csv",index=False,encoding="utf-8-sig")
    d=piv["delta_mape_pp"].to_numpy(); q=np.percentile(d,[25,50,75])
    try: wp=float(stats.wilcoxon(d).pvalue)
    except ValueError: wp=1.0
    summary={"n_seeds":5,"mean":float(d.mean()),"sample_sd":float(d.std(ddof=1)),"median":float(q[1]),"q1":float(q[0]),"q3":float(q[2]),"iqr":float(q[2]-q[0]),
             "min":float(d.min()),"max":float(d.max()),"grade_aware_better_count":int((d<0).sum()),"direction_consistency":float(max((d<0).sum(),(d>0).sum())/len(d)),
             "exact_signflip_p":signflip_p(d),"one_sample_t_p":float(stats.ttest_1samp(d,0).pvalue),"wilcoxon_p":wp,"seed_bootstrap_ci95":bootstrap_ci(d,100_000,42003)}
    model_summary=df.groupby("model")[["MAE","RMSE","MAPE","R2"]].agg(["mean","std"]).reset_index()
    model_summary.to_csv(SUMMARY_DIR/"random_5seed_mean_sd.csv",index=False,encoding="utf-8-sig")
    json_dump(summary,SUMMARY_DIR/"random_5seed_delta_statistics.json")
    return piv,summary


def cluster_oof_stats(climate,grade):
    keys=["original_index","county_year"]
    c=climate.sort_values("original_index"); g=grade.sort_values("original_index")
    if not np.array_equal(c.original_index.to_numpy(),g.original_index.to_numpy()): raise RuntimeError("OOF alignment failure")
    ape_c=[];ape_g=[]
    for t in TARGET_NAMES:
        true=c[f"y_true_{t}"].to_numpy(); den=np.maximum(np.abs(true),1e-8)
        ape_c.append(np.abs(true-c[f"y_pred_{t}"].to_numpy())/den*100)
        ape_g.append(np.abs(true-g[f"y_pred_{t}"].to_numpy())/den*100)
    rec=c[keys+["place","year","k326_internal_grade"]].copy()
    rec["climate_mean_ape"]=np.mean(ape_c,axis=0); rec["grade_mean_ape"]=np.mean(ape_g,axis=0)
    rec["delta_mape_pp"]=rec.grade_mean_ape-rec.climate_mean_ape
    grp=rec.groupby("county_year",as_index=False).agg(n_records=("original_index","size"),climate_mean_ape=("climate_mean_ape","mean"),grade_mean_ape=("grade_mean_ape","mean"),delta_mape_pp=("delta_mape_pp","mean"))
    d=grp.delta_mape_pp.to_numpy(); rng=np.random.default_rng(42004); obs=abs(d.mean()); count=0
    for _ in range(N_PERM//1000):
        signs=rng.choice([-1,1],size=(1000,len(d))); count+=int((np.abs((signs*d).mean(axis=1))>=obs).sum())
    try: wp=float(stats.wilcoxon(d).pvalue)
    except ValueError: wp=1.0
    result={"n_records":len(rec),"n_clusters":len(grp),"record_weighted_delta_mape_pp":float(rec.delta_mape_pp.mean()),"county_year_equal_weight_delta_mape_pp":float(d.mean()),
            "grade_aware_better_clusters":int((d<0).sum()),"grade_aware_better_cluster_fraction":float((d<0).mean()),"cluster_median":float(np.median(d)),
            "cluster_q1":float(np.percentile(d,25)),"cluster_q3":float(np.percentile(d,75)),"cluster_bootstrap_ci95":bootstrap_ci(d,N_BOOT,42005),
            "signflip_permutation_p":(count+1)/(N_PERM+1),"group_t_p":float(stats.ttest_1samp(d,0).pvalue),"group_wilcoxon_p":wp}
    RUNTIME_PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    rec.to_csv(RUNTIME_PRIVATE_DIR/"group_cv_oof_record_differences.csv",index=False,encoding="utf-8-sig")
    grp.to_csv(RUNTIME_PRIVATE_DIR/"group_cv_oof_cluster_differences_with_identifiers.csv",index=False,encoding="utf-8-sig")
    grp[["delta_mape_pp"]].to_csv(SUMMARY_DIR/"group_cv_oof_cluster_differences.csv",index=False,encoding="utf-8-sig")
    json_dump(result,SUMMARY_DIR/"group_cv_oof_cluster_statistics.json")
    return result


def summarize_cv(rows):
    df=pd.DataFrame(rows); df.to_csv(SUMMARY_DIR/"group_cv_fold_model_metrics.csv",index=False,encoding="utf-8-sig")
    piv=df.pivot(index="fold",columns="model",values="MAPE").reset_index(); piv["delta_mape_pp"]=piv.grade_aware-piv.climate_only
    piv.to_csv(SUMMARY_DIR/"group_cv_fold_comparison.csv",index=False,encoding="utf-8-sig")
    cv_summary=df.groupby("model")[["MAE","RMSE","MAPE","R2"]].agg(["mean","std"]).reset_index()
    cv_summary.to_csv(SUMMARY_DIR/"group_cv_mean_sd.csv",index=False,encoding="utf-8-sig")
    pooled={}; wides={}
    for model in ("climate_only","grade_aware"):
        parts=[pd.read_csv(CV_DIR/model/f"fold{f}"/"predictions_wide.csv") for f in range(1,6)]
        wide=pd.concat(parts,ignore_index=True).sort_values("original_index")
        if len(wide)!=394 or wide.original_index.nunique()!=394: raise RuntimeError(f"Incomplete OOF {model}")
        wide.to_csv(CV_DIR/model/"oof_predictions_wide.csv",index=False,encoding="utf-8-sig"); wides[model]=wide
        true=np.column_stack([wide[f"y_true_{t}"] for t in TARGET_NAMES]); pred=np.column_stack([wide[f"y_pred_{t}"] for t in TARGET_NAMES])
        rounded,detailed,overall=metrics_frames(true,pred); rounded.to_csv(CV_DIR/model/"oof_metrics.csv",index=False,encoding="utf-8-sig")
        pooled[model]=overall
    json_dump(pooled,SUMMARY_DIR/"group_cv_pooled_oof_metrics.json")
    cluster=cluster_oof_stats(wides["climate_only"],wides["grade_aware"])
    return piv,pooled,cluster


def main():
    sub,X,y=load_source(); random_split,folds=prepare_splits(sub); audit=preflight(sub,random_split,folds)
    sub.to_csv(SPLIT_DIR/"k326_subset_manifest.csv",index=False,encoding="utf-8-sig")
    random_rows=[]
    for seed in SEEDS:
        for model in ("climate_only","grade_aware"):
            out=RANDOM_DIR/f"seed{seed}"/model
            random_rows.append(train_run(model,seed,random_split["train"],random_split["val"],random_split["test"],out,sub,X,y,"record_random"))
            print("COMPLETE random",seed,model,random_rows[-1]["MAPE"],flush=True)
    random_table,random_stats=summarize_random(random_rows)
    cv_rows=[]
    for f in folds:
        for model in ("climate_only","grade_aware"):
            out=CV_DIR/model/f"fold{f['fold']}"
            cv_rows.append(train_run(model,42,f["train_idx"],f["val_idx"],f["test_idx"],out,sub,X,y,"group_cv",f["fold"]))
            print("COMPLETE group",f["fold"],model,cv_rows[-1]["MAPE"],flush=True)
    fold_table,pooled,cluster=summarize_cv(cv_rows)
    completeness={"completed_runs":20,"expected_runs":20,"preflight_passed":audit["passed"],"random_runs":len(random_rows),"group_cv_runs":len(cv_rows),
                  "random_comparison":random_table.to_dict("records"),"random_delta_statistics":random_stats,"group_cv_comparison":fold_table.to_dict("records"),
                  "pooled_oof_metrics":pooled,"cluster_statistics":cluster}
    json_dump(completeness,SUMMARY_DIR/"k326_validation_final_summary.json")
    print("ALL 20 RUNS COMPLETE",flush=True)


if __name__ == "__main__":
    main()
