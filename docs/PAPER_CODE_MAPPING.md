# Paper-to-Code Mapping

Code repository: [MAJY730/tobacco_chemistry_prediction](https://github.com/MAJY730/tobacco_chemistry_prediction).

This document maps **Multi-Target Prediction of Tobacco Leaf Chemical Composition by Integrating Category Information and Climate Time Series with Group-Based Generalization Assessment** to the shared methods, six core experiments, Figures 1–6, supplementary analyses, and released results. All paths are relative to the repository root.

The repository contains 37 Python source files, 20 CSV files, and 6 PNG files. Commands are given in the [Reproducibility guide](REPRODUCIBILITY.md); input structures and access conditions are described in [Data availability](../DATA_AVAILABILITY.md).

## Shared methods

| Paper location | Related tables or supplementary material | Method | Public code | Authorized inputs and local outputs |
|---|---|---|---|---|
| Methods 2.1–2.4 | Tables 1–2, S1 | Construct 153-day climate inputs, five targets, and record metadata from JSON | `src/core/preprocess/05_build_from_original_json.py` | Reads `less-rules.json` and `config.yaml`; generates O6/W6/W7, y, and `sample_info` |
| Methods 2.2, 2.7 | Table 4, S4 | Encode the raw grade field into IDs for 16 observed categories | `src/core/preprocess/09_prepare_grade_features.py` | Reads `sample_info`; generates grade IDs and mappings |
| Methods 2.5.1 | Table 1 | Fixed seed-42 record-level split: 1,952/418/420 | `src/core/preprocess/06_split_json_dataset.py` | Reads `sample_info` and configuration; generates `split_indices.json` |
| Methods 2.6 | Tables 3–6 | Fit climate-feature and target standardization parameters using training data only | `src/core/preprocess/08_standardize_dataset.py` | Reads X/y and split indices; generates standardized arrays and scalers |
| Methods 2.5.2, 2.6 | Tables 1, 7, S10 | Fixed county-year-disjoint split and corresponding standardization | `src/core/preprocess/10_create_group_split.py`, `src/core/preprocess/11_standardize_group_split.py` | Reads sample information and unstandardized arrays; generates group split indices, standardized arrays, and scalers |
| Methods 2.4 | S1 | Construct, split, and standardize the independently prefiltered more-rules dataset | `src/core/preprocess/12_build_more_rules_dataset.py`, `src/core/preprocess/13_split_standardize_more_rules.py` | Reads authorized `more-rules.json`; generates local artifacts in `processed_more` |
| Methods 2.9.3, 2.10 | Tables 3–7, S3–S11 | Original-scale per-target MAE/RMSE/MAPE/R² and equally weighted five-target averages | `src/core/evaluate/metrics.py` | Accepts inverse-standardized true and predicted values; returns a metrics table |

These arrays, metadata, split indices, and scalers are generated locally by authorized users and are not distributed with the repository.

## Six core experiments

| Experiment | Paper location | Main comparison and reported results | Core code |
|---|---|---|---|
| 01: Climate-only architecture comparison | Results 3.1, Table 3, S2–S3 | Under W7, GRU, CNN+LSTM, Transformer, and TSMixer achieve Avg MAPE values of 22.0792%, 22.0535%, 22.2671%, and 22.0887%, respectively | `src/core/train/train_baselines.py`, `src/core/train/train_transformer_baseline.py`, `src/core/train/train_top_ts_model.py`, and their model modules |
| 02: Category information | Results 3.2, Table 4, Figure 3(A,B), S4 | Climate-only: 22.0887%; Grade mean: 20.1113%; One-hot: 18.1150%; Embedding: 17.8271% | `src/core/category_information/grade_mean_baseline.py`, `src/core/onehot_grade/onehot_grade_tsmixer.py`, `src/core/onehot_grade/train_onehot_grade.py`, and the shared Grade-aware TSMixer branch |
| 03: K326 validation | Results 3.3, Table 4, Figure 3(C,D), S5–S6 | 394 records, 33 counties, 47 county-year groups, and 4 subset categories; fixed record-level five-seed comparison and five-fold Group CV; pooled Avg MAPE values of 23.2201% and 20.3478% for the two models | `src/core/k326_grade_validation/run_k326_validation.py` |
| 04: Tavg × Wind | Results 3.4, Table 5, Figure 4, S7–S8 | Seed-42 results and ten-seed stability of O5/O6/W6/W7; the reported seed-42 W6 value is 18.3474% | 6 scripts in `src/core/wind_2x2/` and the shared `src/core/train/train_top_ts_model.py` |
| 05: Single-target versus Multi-target | Results 3.5, Table 6, Figure 5, S9 | Five ST models versus one MT model; Avg MAPE is 18.1482% for ST and 17.8271% for MT; MT has 179,093 trainable parameters and the five ST models have 894,165 in total | 3 scripts in `src/core/single_target/` and the shared MT training entry |
| 06: County-year generalization | Results 3.6, Table 7, Figure 6, S10–S11 | Fixed county-year-disjoint split and five-fold OOF evaluation; pooled Avg MAPE is 21.2658%, 21.2857%, and 21.3365% for Grade-GRU, Grade-Transformer, and GA-Wind-TSMixer, respectively; paired statistics use 208 county-year groups | Preprocessing scripts 10/11, `src/core/train/stage9_train_compact.py`, `src/core/group_cv/run_group_cv.py`, and `src/core/group_cv/analyze_oof_cluster_statistics.py` |

The Climate-only TSMixer in Experiments 01 and 02, and the W7 Grade-aware TSMixer in Experiments 02 and 04, reuse their corresponding model and training settings. No duplicate model or training files are needed. Model modules retain additional optional branches; the main comparisons use the entries and commands specified here and in the reproducibility guide.

## Twenty released CSV files

All files below are included in the repository. Item 7 is a deidentified figure input; the other 19 files contain aggregate results.

| No. | Experiment | Repository-relative path | Results supported |
|---|---|---|---|
| 1 | 01 | `src/core/results/tables/S3_Complete_five_target_metrics.csv` | Four models, five targets, four metrics, and Avg |
| 2 | 02 | `src/core/results/tables/S4_Panel_B_Grade_representation_comparison.csv` | Aggregate comparison of Climate-only, One-hot, and Embedding |
| 3 | 02 | `src/core/results/tables/grade_mean_metrics.csv` | Grade mean metrics for five targets and Avg |
| 4 | 03 | `src/core/k326_grade_validation/summary/random_5seed_comparison.csv` | Five-seed comparison of two models under the fixed split |
| 5 | 03 | `src/core/k326_grade_validation/group_cv/climate_only/oof_metrics.csv` | K326 Climate-only pooled OOF metrics |
| 6 | 03 | `src/core/k326_grade_validation/group_cv/grade_aware/oof_metrics.csv` | K326 Grade-aware pooled OOF metrics |
| 7 | 03 | `src/core/k326_grade_validation/summary/group_cv_oof_cluster_differences.csv` | Figure 3 input with 47 rows of `delta_mape_pp` and no county-year identifiers |
| 8 | 04 | `src/core/wind_2x2/multiseed/summary/ten_seed_four_configuration_avg_mape.csv` | Avg MAPE for ten seeds and four configurations |
| 9 | 04 | `src/core/wind_2x2/multiseed/summary/configuration_cross_seed_summary.csv` | Cross-seed summaries of the four configurations |
| 10 | 04 | `src/core/wind_2x2/multiseed/summary/effect_cross_seed_statistics.csv` | Cross-seed paired effects and statistical results |
| 11 | 05 | `src/core/single_target/results/seed42/summary/single_vs_multi_overall.csv` | Overall ST/MT metric comparison |
| 12 | 05 | `src/core/single_target/results/seed42/summary/single_vs_multi_per_target.csv` | Per-target ST/MT metric comparison |
| 13 | 05 | `src/core/single_target/results/seed42/summary/parameter_comparison.csv` | Parameter counts for one MT model and five ST models |
| 14 | 05 | `src/core/single_target/results/seed42/statistical_validation/overall_cluster_statistics.csv` | Overall ST−MT county-year cluster statistics |
| 15 | 05 | `src/core/single_target/results/seed42/statistical_validation/per_target_cluster_statistics.csv` | Per-target ST−MT county-year cluster statistics |
| 16 | 06 | `src/core/group_cv/summary/evaluation_comparison.csv` | Model performance under different evaluation splits |
| 17 | 06 | `src/core/group_cv/summary/fold_model_comparison.csv` | Model comparison across five folds |
| 18 | 06 | `src/core/group_cv/summary/pooled_oof_metrics_all_models.csv` | Aggregate pooled OOF metrics for three models |
| 19 | 06 | `src/core/group_cv/summary/group_weighted_metrics.csv` | County-year-equal aggregate metrics |
| 20 | 06 | `src/core/group_cv/summary/oof_cluster_statistics/group_cv_oof_cluster_statistics.csv` | Paired model statistics across 208 county-year groups |

K326 grouped tables with county-year identifiers and record-level differences are written locally to `src/core/k326_grade_validation/runtime_private/`, which is excluded by `.gitignore`. Record-level predictions, checkpoints, logs, and detailed outputs from other experiments are also local runtime artifacts.

## Figures 1–6 and plotting entries

| Figure | Current PNG | Plotting code and inputs |
|---|---|---|
| 1 | `figures/figure1_data_framework.png` | Data hierarchy and split schematic; no Python redraw entry is provided |
| 2 | `figures/figure2_model_architecture.png` | Model architecture schematic; no Python redraw entry is provided |
| 3 | `figures/figure3/Figure3_category_information_K326_composite.png` | `figures/figure3/figure3_category_k326_frontiers_composite.py`; A–C use reported aggregate values, and D reads 47 deidentified county-year differences |
| 4 | `figures/figure4/Figure4_Tavg_Wind_2x2_composite.png` | `figures/figure4/figure4_tavg_wind_frontiers_composite.py`; reads the ten-seed four-configuration results and cross-seed effect statistics |
| 5 | `figures/figure5/Figure5_single_vs_multi_composite.png` | `figures/figure5/figure5_single_vs_multi_frontiers_composite.py`; reads per-target and overall county-year cluster statistics and parameter comparisons |
| 6 | `figures/figure6/Figure6_county_year_generalization_composite.png` | `figures/figure6/figure6_group_cv_frontiers_composite.py`; reads evaluation-setting, fold, pooled OOF, county-year-equal, and paired statistical summaries |

Use the four `*_frontiers_composite.py` entries in the table to redraw the current PNG files.

The inputs for Figures 3–6 are included among the 20 released CSV files. The scripts use repository-relative paths. Rendering may vary with fonts and Matplotlib versions.

## Additional supplementary analyses

- **S1:** Construction, splitting, standardization, and training entries are provided for the prefiltered more-rules dataset. The complete upstream filtering procedure and a standalone entry for evaluation after excluding exact training–test duplicates are not included.
- **S2:** Architectures and configurations are defined in the model modules and training entries. Transformer has 72,069 trainable parameters; its positional-encoding buffer is counted separately.
- **S3–S11:** Released CSV files support checking the main results of the six core experiments, including pooled OOF metrics for both K326 models. Not every complete Supplementary Material table is provided as a separate CSV.
- **S12:** Grade mean has a standalone entry. Complete dedicated entries for Grade-only Embedding, Shuffled-grade, and other category diagnostics are not included.

## Evaluation and output definitions

Five-fold OOF evaluation gives each record one out-of-fold prediction. The formal paired comparisons between models use county-years as the statistical units; fold means and standard deviations describe variation across splits.

The fixed record-level and fixed county-year group reference values in `evaluation_comparison.csv` are supplied as source-code constants. Running five-fold Group CV does not recompute these two columns. Corresponding training commands are given in the reproducibility guide.

The repository distributes 19 aggregate CSV files and one deidentified figure input. It does not distribute sample-level metadata, grade IDs/mappings, splits, scalers, record-level predictions or OOF outputs, checkpoints, or detailed outputs linkable to county-years.
