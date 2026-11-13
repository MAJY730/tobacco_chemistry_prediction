# Predicting Tobacco Leaf Chemical Composition from Climate Time Series and Category Information

Code repository: [MAJY730/tobacco_chemistry_prediction](https://github.com/MAJY730/tobacco_chemistry_prediction).

This repository provides the core preprocessing, modeling, training, evaluation, statistical analysis, and plotting code for **Multi-Target Prediction of Tobacco Leaf Chemical Composition by Integrating Category Information and Climate Time Series with Group-Based Generalization Assessment**.

The study examines how sample-level category information, climate-variable configurations, model architecture, multi-target parameter sharing, and evaluation splits relate to the prediction of five chemical constituents when multiple tobacco leaf samples within a county-year share the same growing-season climate sequence.

## Data and prediction task

The main analysis includes 2,790 tobacco leaf sample records collected from 72 tobacco-growing counties in Yunnan Province, China, during 2013–2018, representing 208 county-year climatic backgrounds. Each climate input covers 153 days, from May 1 to September 30.

- W7 climate input order: `Tmax`, `Tmin`, `Tavg`, `Rain`, `Sunshine`, `Pressure`, `Wind`.
- Five-target order: `total_sugar`, `reducing_sugar`, `nicotine`, `total_nitrogen`, `potassium`.
- The raw `grade` field contains 16 observed categories, including grade labels and grade–variety combination labels.
- The main model, GA-Wind-TSMixer, combines a 64-dimensional climate representation with an 8-dimensional learnable category embedding to predict all five targets jointly.

Tobacco leaf chemical measurements are subject to third-party restrictions on data use and sharing and are not distributed with this repository. Daily meteorological data were obtained from the WheatA agricultural meteorological big data platform. Authorized users can regenerate the processed arrays, data splits, standardization parameters, and experimental outputs by following the instructions. Access conditions and input structures are described in [Data availability](DATA_AVAILABILITY.md).

## Repository contents

| Directory or file | Purpose |
|---|---|
| `requirements.txt` | 8 direct dependencies with bounded version ranges for the shared software environment |
| `src/core/config.yaml` | Sequence length, climate variables, prediction targets, and fixed-split configuration |
| `src/core/preprocess/` | Main and more-rules dataset construction, grade encoding, record-level and county-year group splits, and standardization fitted on training data |
| `src/core/models/` | Baseline time-series and grade-aware model definitions |
| `src/core/train/` | Shared training entries and training entries for the fixed group split and sensitivity analyses |
| `src/core/evaluate/` | Evaluation metrics for the five targets |
| `src/core/category_information/` | Grade mean baseline for Experiment 02 |
| `src/core/onehot_grade/` | One-hot model and training entry for Experiment 02 |
| `src/core/k326_grade_validation/` | K326 five-seed validation and group cross-validation for Experiment 03 |
| `src/core/wind_2x2/` | Tavg × Wind factorial experiment, multi-seed training, and statistical analysis for Experiment 04 |
| `src/core/single_target/` | Single-target training and ST/MT comparisons for Experiment 05 |
| `src/core/group_cv/` | Five-fold county-year Group CV and paired statistics for Experiment 06 |
| `src/core/results/tables/` and experiment-specific `summary/` or result directories | Released aggregate results and the deidentified figure input |
| `figures/` | 6 PNG files for Figures 1–6 and plotting code for Figures 3–6 |
| `docs/` | Paper-to-code mapping and reproducibility guide |
| `.gitignore` | Rules excluding restricted data and local runtime artifacts from version control |

The repository contains **37 Python files, 20 CSV files, and 6 PNG files**. The Python files comprise 33 core source files and 4 plotting scripts. The plotting entries for Figures 3–6 are the `*_frontiers_composite.py` scripts.

Of the 20 CSV files, 19 contain model-level or statistical aggregates. The remaining file is a deidentified input for Figure 3 containing 47 rows of `delta_mape_pp`, without county-year identifiers. Figures 1 and 2 are provided as `figures/figure1_data_framework.png` and `figures/figure2_model_architecture.png`. The current composite figures, plotting code, and aggregate inputs for Figures 3–6 are included.

## Software environment and installation

The validated reference software environment, checked for installation and compatibility on Windows/CPU, is:

| Software | Reference version |
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

Python 3.11 is recommended. `requirements.txt` specifies bounded version ranges, so resolved versions may differ slightly from the reference versions. XGBoost is used only in an optional baseline branch and is not required for the six core experiments.

Run the following commands in Windows PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Environment validation covered installation, dependency imports, core APIs, and model-construction compatibility. All 37 Python files passed static syntax checks. These software checks did not include rerunning the complete data-dependent experiments.

## Data preparation and execution order

All commands and source paths are relative to the repository root. First read the input specification, then place an authorized `less-rules.json` file with the required record order in the local `src/core/data/raw/` directory. The construction script searches recursively by filename; keep only one input file with the corresponding name within the search scope.

Run the main preprocessing steps in this order:

```powershell
python src/core/preprocess/05_build_from_original_json.py
python src/core/preprocess/09_prepare_grade_features.py
python src/core/preprocess/06_split_json_dataset.py
python src/core/preprocess/08_standardize_dataset.py
```

The fixed record-level split uses `RandomState(42)` and produces 1,952 training, 418 validation, and 420 test records for the study dataset. Climate-feature and target means and standard deviations are fitted using training records only; the same parameters are applied to validation and test data. Reproducing the same record split also requires the original record order and the same set of records passing validation.

Experiment prerequisites, commands, and output locations are listed in the [Reproducibility guide](docs/REPRODUCIBILITY.md).

## Six core experiments

| Experiment | Research question and main code | Paper location |
|---|---|---|
| 01: Climate-only architecture comparison | Compare GRU, CNN+LSTM, Transformer, and TSMixer under the common W7 input using three shared training entries in `src/core/train/` | Results 3.1, Table 3, S2–S3 |
| 02: Category information | Compare Climate-only, Grade mean, Climate + One-hot, and Climate + Embedding using the Grade mean, One-hot, and shared TSMixer entries | Results 3.2, Table 4, Figure 3(A,B), S4 |
| 03: K326 validation | Fixed record-level five-seed validation and county-year Group CV on the K326 subset, using `k326_grade_validation/run_k326_validation.py` | Results 3.3, Figure 3(C,D), S5–S6 |
| 04: Tavg × Wind | Seed-42 comparison and ten-seed stability analysis of O5/O6/W6/W7, implemented in `wind_2x2/` | Results 3.4, Table 5, Figure 4, S7–S8 |
| 05: Single-target versus Multi-target | Compare prediction errors and parameter-sharing efficiency of five independent single-target models and one multi-target model, implemented in `single_target/` | Results 3.5, Table 6, Figure 5, S9 |
| 06: County-year generalization | Fixed county-year-disjoint split and five-fold Group CV, using preprocessing scripts 10/11, `stage9_train_compact.py`, and `group_cv/` | Results 3.6, Table 7, Figure 6, S10–S11 |

The four architectures in Experiment 01 share W7 inputs, a fixed split, and evaluation definitions. GRU/CNN+LSTM use Adam, at most 200 epochs, early-stopping patience of 20, and no learning-rate scheduler. Transformer/TSMixer use AdamW, at most 300 epochs, early-stopping patience of 30, and ReduceLROnPlateau.

Evaluation is performed on the original target scale after inverse standardization. MAE, RMSE, MAPE, and R² are reported for each target, together with their equally weighted five-target averages (Avg). The MAPE denominator is `max(abs(y_true), 1e-8)`.

## Released results and reproducibility scope

Aggregate results are stored in `src/core/results/tables/` and experiment-specific `summary/` or result directories. The 20 CSV files support checking the main results of all six core experiments. The two K326 model-level `oof_metrics.csv` files directly report pooled Avg MAPE values of 23.2201% for Climate-only and 20.3478% for Grade-aware. A complete file index is provided in the [Paper-to-code mapping](docs/PAPER_CODE_MAPPING.md).

Figures 3–6 can be redrawn using the aggregate inputs supplied with the repository, without obtaining the underlying records or retraining the models. Rendering may vary with fonts and Matplotlib versions; software compatibility checks do not establish pixel-identical rendering across environments.

Full preprocessing, training, and record-level statistical analyses require authorized data. Raw records, sample-level metadata, processed record-level arrays, grade mappings, splits, scalers, sample-level predictions, record-level OOF predictions, checkpoints, and details linkable to county-years are not distributed. K326 grouped tables with identifiers and record-level differences are stored locally in `runtime_private/`, which is excluded by `.gitignore`. Only reviewed source code, figures, and aggregate results should be released after a local run.

Supplementary analysis coverage is as follows. The S1 more-rules branch uses authorized inputs already filtered according to the study definition; the repository does not include the complete upstream filtering procedure or a standalone entry for the exact-duplicate exclusion analysis. Complete dedicated entries for the S12 Grade-only Embedding and Shuffled-grade diagnostics are not included. These analyses remain reported in the paper and Supplementary Material.

## Citation and license

Software citation information is provided in `CITATION.cff` at the repository root, with Yunwei Zhang, Jingyi Ma, and Kuai Dai as citation authors, in that order. The file describes the software itself. A software version, release date, and code DOI will be added when available. The paper citation and paper DOI will be added once publication details are available.

The software source code, configuration, and accompanying software documentation are licensed under the MIT License; see `LICENSE` at the repository root. Copyright is held by Yunwei Zhang, Jingyi Ma, and Kuai Dai, with the year 2026. MIT permits use, copying, modification, redistribution, and commercial use, subject to retaining the copyright and permission notice.

The software license does not grant access to or redistribution rights for restricted research data. Use of paper figures and released aggregate results is subject to their respective source and rights notices; these materials are not automatically covered by the code's MIT License.
