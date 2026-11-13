# Reproducibility Workflow and Scope

Code repository: [MAJY730/tobacco_chemistry_prediction](https://github.com/MAJY730/tobacco_chemistry_prediction).

Run all commands in this guide from the repository root. The repository contains 37 Python source files, 20 released CSV files, and 6 PNG files. Nineteen CSV files contain aggregate results; the remaining file contains deidentified differences used in Figure 3. The supplied files support reading aggregate results and redrawing Figures 3–6. Full preprocessing, training, and record-level statistical analyses require authorized research data.

## 1. Software environment

Python 3.11 is recommended. The validated Windows/CPU reference software environment, checked for installation and compatibility, is:

| Software | Version |
|---|---|
| Python | 3.11.16 |
| NumPy | 1.26.4 |
| pandas | 2.2.3 |
| PyTorch | 2.6.0 |
| SciPy | 1.14.1 |
| scikit-learn | 1.5.2 |
| Matplotlib | 3.9.4 |
| openpyxl | 3.1.5 |
| PyYAML | 6.0.3 |

`requirements.txt` uses bounded version ranges, so resolved versions may differ slightly. Install the dependencies with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

XGBoost is used only in an optional baseline branch and is not required for the six core experiments. The dependency file does not pin a CUDA-specific wheel or toolkit.

The reference environment has passed installation, dependency-import, core-API, model-construction, and architecture-compatibility checks. All 37 Python files passed static syntax checks. This validation did not include rerunning all data-dependent experiments and does not establish complete reproduction of the scientific results.

## 2. Authorized data and record order

Place an authorized `less-rules.json` conforming to the input specification in the local `src/core/data/raw/` directory. The S1 more-rules branch additionally requires `more-rules.json`, independently prefiltered according to the paper's definition. Construction scripts search by filename; keep only one file with each corresponding name within the search scope. See [Data availability](../DATA_AVAILABILITY.md) for fields and access conditions.

The source weather-column order is `Tmax/Tmin/Tavg/Rain/Pressure/Sunshine/Wind`; the W7 modeling order is `Tmax/Tmin/Tavg/Rain/Sunshine/Pressure/Wind`. The five targets are selected from `chemical` indices `[0, 1, 2, 5, 4]`, in the order total sugar, reducing sugar, nicotine, total nitrogen, and potassium. The time window is May 1 to September 30, covering 153 days.

Reproducing the study's fixed split requires the original record order and the same set of records passing validation. Reordering the input changes record identities under the same seed and split proportions. Arrays, grade IDs, split indices, and original indices must remain aligned.

## 3. Shared preprocessing

```powershell
python src/core/preprocess/05_build_from_original_json.py
python src/core/preprocess/09_prepare_grade_features.py
python src/core/preprocess/06_split_json_dataset.py
python src/core/preprocess/08_standardize_dataset.py
```

All four entries can be run without arguments:

1. Script 05 constructs unstandardized O6/W6/W7 arrays, five-target y, and `sample_info`. Its first positional argument changes the filename to search for; it does not directly specify a complete input path.
2. Script 09 generates category IDs and mappings from the grade field. The main dataset contains 16 observed categories.
3. Script 06 uses `RandomState(42)` to create a record-level split with 70% training, 15% validation, and the remainder for testing, sorting indices within each subset. The study split contains 1,952/418/420 records.
4. Script 08 fits X/y means and population standard deviations using training records only and applies those parameters to validation and test data. Standard deviations below `1e-10` are replaced by 1.

Outputs are written to `src/core/data/processed/`, including unstandardized and standardized arrays, sample information, category mappings, split indices, and scalers. Keep these outputs within the authorized data environment.

## 4. Experiment 01: Four architectures under the common W7 input

After shared preprocessing, run:

```powershell
python src/core/train/train_baselines.py --model gru --input_type wind7 --seed 42 --epochs 200 --batch_size 32 --lr 0.001
python src/core/train/train_baselines.py --model cnn_lstm --input_type wind7 --seed 42 --epochs 200 --batch_size 32 --lr 0.001
python src/core/train/train_transformer_baseline.py --model transformer --input_type wind7 --variant default --seed 42 --epochs 300 --batch_size 32 --lr 0.001 --weight_decay 0.0001 --patience 30
python src/core/train/train_top_ts_model.py --model tsmixer --input_type wind7 --variant default --seed 42 --epochs 300 --batch_size 32 --lr 0.001 --weight_decay 0.0001 --patience 30
```

GRU/CNN+LSTM use Adam, early-stopping patience of 20, and no learning-rate scheduler. Transformer/TSMixer use AdamW, early-stopping patience of 30, and ReduceLROnPlateau with a factor of 0.5 and patience of 10. Models minimize MSE in standardized target space and are selected by the lowest validation MSE. Test predictions are inverse-standardized before evaluation with the shared metrics module.

| Model | Registered trainable parameters | Reported Avg MAPE |
|---|---|---|
| GRU | 14,341 | 22.0792% |
| CNN+LSTM | 40,517 | 22.0535% |
| Transformer | 72,069 | 22.2671% |
| TSMixer | 173,968 | 22.0887% |

Transformer's 9,792 positional-encoding elements are a buffer and are excluded from the trainable parameter count. Released results are in `src/core/results/tables/S3_Complete_five_target_metrics.csv`. Sample-level predictions and checkpoints generated by training remain local.

## 5. Experiment 02: Category information

Climate-only reuses the TSMixer setting from Experiment 01. Grade mean uses only original-scale y, grade IDs, and the fixed split. It calculates the five-target mean for training records in each category, then predicts each test record using its category mean. A category absent from training causes an error; no global-mean fallback is used.

```powershell
python src/core/category_information/grade_mean_baseline.py --y src/core/data/processed/y.npy --grade_ids src/core/data/processed/grade_ids.npy --split src/core/data/processed/split_indices.json --output src/core/results/tables/grade_mean_metrics.csv
python src/core/onehot_grade/train_onehot_grade.py
python src/core/train/train_top_ts_model.py --model grade_tsmixer --input_type wind7 --variant default --seed 42 --epochs 300 --batch_size 32 --lr 0.001 --weight_decay 0.0001 --patience 30
```

The One-hot entry has no CLI. Its source-defined settings are seed 42, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, at most 300 epochs, early-stopping patience of 30, and ReduceLROnPlateau with a factor of 0.5 and patience of 10. Training uses standardized-target MSE; the model with the lowest validation loss is selected and evaluated after inverse standardization.

One-hot concatenates a 64-dimensional climate representation with a fixed 16-dimensional category vector and uses an 80→64→5 regression head. Embedding uses an 8-dimensional learnable category representation and a 72→64→5 head. Their registered trainable parameter counts are 179,477 and 179,093, respectively.

The One-hot entry writes only aggregate metrics to `src/core/onehot_grade/results/seed42/tables/onehot_grade_tsmixer_seed42_metrics.csv`; it does not output sample-level predictions or checkpoints. The shared Embedding entry generates the W7 record-level predictions needed by later experiments. Keep these predictions in the authorized local environment.

The reported Avg MAPE values for the four settings are 22.0887%, 20.1113%, 18.1150%, and 17.8271%, respectively. The released `src/core/results/tables/S4_Panel_B_Grade_representation_comparison.csv` and `grade_mean_metrics.csv` support checking the aggregate results. The Grade mean entry covers only the corresponding S12 baseline, not all category diagnostics.

## 6. Experiment 03: K326 validation

```powershell
python src/core/k326_grade_validation/run_k326_validation.py
```

This entry has no CLI. It reads the main dataset's unstandardized W7/y, `sample_info`, grade IDs/mappings, and fixed split. It selects 394 K326 records from 33 counties and 47 county-year groups and creates 4 category IDs internal to the subset.

The record-level split remains fixed while Climate-only and Grade-aware models are retrained using seeds 42, 123, 456, 789, and 2024. The entry also runs five-fold county-year Group CV. Each training run fits standardization parameters using its training indices only, with a standard-deviation handling threshold of `1e-8`.

K326 training uses AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size 32, at most 300 epochs, early-stopping patience of 30, ReduceLROnPlateau with a factor of 0.5 and patience of 10, and standardized-target MSE. Model selection uses the lowest validation loss.

Released results include:

- `src/core/k326_grade_validation/summary/random_5seed_comparison.csv`: five-seed comparison under the fixed split.
- `src/core/k326_grade_validation/group_cv/climate_only/oof_metrics.csv`: pooled Avg MAPE of 23.2201%.
- `src/core/k326_grade_validation/group_cv/grade_aware/oof_metrics.csv`: pooled Avg MAPE of 20.3478%.
- `src/core/k326_grade_validation/summary/group_cv_oof_cluster_differences.csv`: deidentified Figure 3 input containing only 47 rows of `delta_mape_pp`.

Complete grouped tables with county-year identifiers and record-level differences are written to `src/core/k326_grade_validation/runtime_private/`, which is excluded by `.gitignore`. Subset record lists, fold membership, and record-level predictions are not distributed.

## 7. Experiment 04: Tavg × Wind

The four climate configurations use the following input orders:

| Configuration | Climate variables |
|---|---|
| O5 | Tmax, Tmin, Rain, Sunshine, Pressure |
| O6 | Tmax, Tmin, Tavg, Rain, Sunshine, Pressure |
| W6 | Tmax, Tmin, Rain, Sunshine, Pressure, Wind |
| W7 | Tmax, Tmin, Tavg, Rain, Sunshine, Pressure, Wind |

All settings use Grade-aware TSMixer with the same category inputs and different climate channels. Run the entries in this order:

```powershell
python src/core/wind_2x2/prepare_original5.py
python src/core/wind_2x2/train_original5_grade_tsmixer.py
python src/core/train/train_top_ts_model.py --model grade_tsmixer --input_type original6 --variant default --seed 42
python src/core/train/train_top_ts_model.py --model grade_tsmixer --input_type wind6 --variant default --seed 42
python src/core/train/train_top_ts_model.py --model grade_tsmixer --input_type wind7 --variant default --seed 42
python src/core/wind_2x2/analyze_2x2_effects.py
python src/core/wind_2x2/statistical_validation_2x2.py
python src/core/wind_2x2/run_multiseed_2x2.py
python src/core/wind_2x2/summarize_multiseed_2x2.py
```

The shared training entry generates seed-42 record-level predictions for local use by the seed-42 analyses and ten-seed summaries. The multi-seed script trains all four configurations for 9 additional seeds, giving 36 training runs. The summarizer reuses seed-42 outputs without retraining that seed.

| Configuration | Seed-42 Avg MAPE | Ten-seed Avg MAPE, mean ± sample standard deviation |
|---|---|---|
| O5 | 17.9018% | 18.2257 ± 0.2762% |
| O6 | 18.3193% | 18.1397 ± 0.1250% |
| W6 | 18.3474% | 18.2190 ± 0.2539% |
| W7 | 17.8271% | 18.1528 ± 0.2587% |

S7 reports the complete seed-42 results; S8 reports ten-seed stability. Configuration differences are defined as the latter minus the former configuration, so a negative value indicates lower error for the latter. The interaction is `(W7 − W6) − (O6 − O5)`.

Released aggregates are stored in `src/core/wind_2x2/multiseed/summary/`: `ten_seed_four_configuration_avg_mape.csv`, `configuration_cross_seed_summary.csv`, and `effect_cross_seed_statistics.csv`.

## 8. Experiment 05: Single-target versus Multi-target

First retain the local Multi-target predictions from the seed-42 Grade-aware W7 model in Experiment 02, then run:

```powershell
python src/core/single_target/train_single_target.py
python src/core/single_target/analyze_single_vs_multi.py
python src/core/single_target/statistical_validation.py
```

None of these three scripts has a CLI. The five ST models are constructed independently, with seed 42 reset for each target. Each ST model has 178,833 registered trainable parameters, giving 894,165 across five models; MT has 179,093. Reported Avg MAPE is 17.8271% for MT and 18.1482% for ST, with an ST−MT difference of +0.3211 percentage points.

The comparison script uses unstandardized y as the common ground truth and reads predictions from five ST models and one MT model to generate per-target, overall, and parameter-count comparisons. The statistical script uses `sample_info` to perform 100,000 cluster sign-flips, 10,000 cluster bootstraps, a one-sample t-test, a Wilcoxon test, and Holm correction across 149 county-year groups.

Released files include `single_vs_multi_overall.csv`, `single_vs_multi_per_target.csv`, and `parameter_comparison.csv` in `src/core/single_target/results/seed42/summary/`, plus `overall_cluster_statistics.csv` and `per_target_cluster_statistics.csv` in `statistical_validation/`. Record-level predictions and error details remain local.

## 9. Experiment 06: County-year generalization evaluation

Create the fixed county-year-disjoint split and train the three Grade-aware models:

```powershell
python src/core/preprocess/10_create_group_split.py
python src/core/preprocess/11_standardize_group_split.py
python src/core/train/stage9_train_compact.py --model grade_gru --input_type wind7 --data_source group --seed 42
python src/core/train/stage9_train_compact.py --model grade_transformer --input_type wind7 --data_source group --seed 42
python src/core/train/stage9_train_compact.py --model grade_tsmixer --input_type wind7 --data_source group --seed 42
```

The same training entry supports the additional O6 Climate-only baseline comparison under the fixed group split:

```powershell
python src/core/train/stage9_train_compact.py --model gru --input_type original6 --data_source group --seed 42
python src/core/train/stage9_train_compact.py --model cnn_lstm --input_type original6 --data_source group --seed 42
```

Run five-fold county-year Group CV and OOF statistical analysis with:

```powershell
python src/core/group_cv/run_group_cv.py
python src/core/group_cv/analyze_oof_cluster_statistics.py
```

The Group CV entry trains the three main Grade-aware models by default and also supports `--prepare-only` and `--models`. All three models share the outer folds; each county-year group enters the outer test set once. Fold seeds for the inner validation split are 43–47, and the training seed is 42. Each fold's scalers are fitted using inner training records only.

OOF predictions cover all 2,790 records. Pooled metrics weight records equally, whereas county-year-equal metrics give equal weight to the 208 groups. Fold means and standard deviations describe variation across splits and are not treated as independent statistical replicates. The paired statistical entry requires complete OOF predictions from all three models and should run after all models finish.

| Model | Pooled OOF Avg MAPE |
|---|---|
| Grade-GRU | 21.2658% |
| Grade-Transformer | 21.2857% |
| GA-Wind-TSMixer | 21.3365% |

The five released CSV files are in `src/core/group_cv/summary/` and its `oof_cluster_statistics/` subdirectory; complete paths are listed in the paper-to-code mapping. Record-level OOF predictions and training-generated `best_checkpoint.pth` files remain local.

`run_group_cv.py` checks architectures against the source-defined `EXPECTED_ARCHITECTURES` and does not require an external reference checkpoint. The primary OOF evaluation retains 16 outer-test records whose categories are absent from the corresponding inner training sets. A separate descriptive sensitivity result excludes those records.

The fixed-setting reference values in `evaluation_comparison.csv` are source-code constants. Running five-fold Group CV does not recompute the fixed record-level or fixed county-year group reference columns. Run the corresponding training entries to check those settings locally.

## 10. S1 more-rules sensitivity analysis

Obtain authorized `more-rules.json`, already filtered according to the paper's definition, and prepare the main dataset's grade mapping before running:

```powershell
python src/core/preprocess/12_build_more_rules_dataset.py
python src/core/preprocess/13_split_standardize_more_rules.py
python src/core/train/stage9_train_compact.py --model grade_tsmixer --input_type wind7 --data_source more_rules --tag_prefix stage9_more --seed 42
python src/core/train/stage9_train_compact.py --model grade_gru --input_type wind7 --data_source more_rules --tag_prefix stage9_more --seed 42
python src/core/train/stage9_train_compact.py --model grade_transformer --input_type wind7 --data_source more_rules --tag_prefix stage9_more --seed 42
```

The study's more-rules dataset has 2,663 records, with 1,864/399/400 training/validation/test records. Scripts 12/13 use the prefiltered JSON and do not implement the complete upstream filtering from less-rules. Main and more-rules category ID orders differ; use the corresponding mappings. The S1 exact-duplicate exclusion evaluation has no standalone public entry.

## 11. Metric definitions and Figures 1–6

The shared `src/core/evaluate/metrics.py` calculates original-scale per-target MAE, RMSE, MAPE, and R², along with equally weighted five-target averages. The MAPE denominator is `max(abs(y_true), 1e-8)`; R² is set to 0 when `SS_tot < 1e-10`. Returned values are rounded to four decimal places.

Grade mean and some statistical scripts retain local implementations without rounding. Compare outputs at the precision used by the corresponding report. The Single-target local R² implementation does not use the shared module's low-variance branch. All dependent files must preserve record order, category IDs, and index alignment.

Figures 1 and 2 are provided as:

- `figures/figure1_data_framework.png`
- `figures/figure2_model_architecture.png`

These are existing schematics; the repository provides no Python redraw entries for them. Figures 3–6 can be redrawn directly from the aggregate inputs included in the repository:

```powershell
python figures/figure3/figure3_category_k326_frontiers_composite.py
python figures/figure4/figure4_tavg_wind_frontiers_composite.py
python figures/figure5/figure5_single_vs_multi_frontiers_composite.py
python figures/figure6/figure6_group_cv_frontiers_composite.py
```

The current outputs are the `*_composite.png` files in the Figure 3–6 directories. The plotting scripts also generate PDF/SVG files. Input paths, columns, row counts, and key values have been checked statically. Rendering can vary with fonts and Matplotlib versions; reference-environment compatibility checks do not establish pixel-identical rendering across environments.

Use the four commands above to redraw the current figures.

## 12. Local execution and public release

This repository provides core method code, main aggregate results, and plotting code. Full data processing and model training require authorized research data.

Sample-level predictions, record-level OOF predictions, location/year/category metadata, grouped details, fitted scalers, checkpoints, and logs generated during execution remain within the local authorized environment and are not distributed with the repository. `.gitignore` excludes identified restricted inputs and local runtime artifacts while retaining the 20 CSV files supplied with the repository.

K326 grouped tables with county-year identifiers and record-level differences are written to `src/core/k326_grade_validation/runtime_private/`. The released `summary/group_cv_oof_cluster_differences.csv` contains only 47 rows of `delta_mape_pp`, without county-year identifiers.

Repeating a run with the same model, input, seed, or tag may overwrite local results. Some summary and plotting scripts also update aggregate results or figures supplied with the repository. Before rerunning, preserve the original released files and save newly generated outputs separately from the results reported in the paper.
