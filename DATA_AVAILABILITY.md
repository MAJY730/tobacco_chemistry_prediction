# Data Availability and Input Specification

Code repository: [MAJY730/tobacco_chemistry_prediction](https://github.com/MAJY730/tobacco_chemistry_prediction).

The tobacco leaf chemical measurement data were provided by Yunnan Tobacco Leaf Company and are subject to third-party restrictions on use and sharing. The underlying research records are not distributed with this code repository. Daily meteorological data were obtained from the WheatA agricultural meteorological big data platform. Released research results are available in the paper, Supplementary Material, and aggregate CSV files supplied with this repository.

Reasonable data requests may be directed to the corresponding author within the limits of the existing permissions. Any sharing is subject to the necessary authorization from the data provider. This repository does not grant access to or redistribution rights for the underlying records and does not promise unrestricted access to the complete raw dataset.

**Data access contact:** Kuai Dai, corresponding author; email: [daikuai520@163.com](mailto:daikuai520@163.com).

## Included and excluded materials

| Material | Repository scope |
|---|---|
| Preprocessing, modeling, training, evaluation, statistical analysis, and plotting code | Included |
| Model-level and statistical aggregate results | 19 CSV files covering the main results of the six core experiments |
| Deidentified input for Figure 3 | 1 CSV containing only 47 rows of `delta_mape_pp`, without county-year or sample identifiers |
| Figures 1–6 | 6 PNG files; plotting source code is also provided for Figures 3–6 |
| Raw JSON, sample-level tobacco chemical measurements, record-level weather sequences, and sample information | Not distributed |
| Processed arrays, grade IDs/mappings, splits, scalers, sample-level predictions, record-level OOF predictions, details linkable to county-years, and checkpoints | Not distributed |

Released aggregate results are organized by experiment under `src/core/`. The public CSV files contain summaries calculated from the research data, without the underlying sample-level observations or record-level predictions. Individual files are listed in the [Paper-to-code mapping](docs/PAPER_CODE_MAPPING.md).

## Inputs required from authorized users

The main dataset is constructed from `less-rules.json`. The S1 more-rules sensitivity analysis additionally requires `more-rules.json`, already filtered according to the study definition. Place authorized input files in the local `src/core/data/raw/` directory. Both construction scripts search by filename under `src/`; keep only one input file with the corresponding name within that search scope.

The top-level JSON object must be a list of records. Each record contains the following fields. This specification defines the structure only and provides no actual records or example values.

| Field | Type and meaning | Use |
|---|---|---|
| `year` | Study year of the record | County-year grouping and record metadata |
| `place` | County or location identifier | Defines a county-year group together with the year |
| `grade` | Raw category label, potentially including a variety suffix | Category ID and mapping generation |
| `chemical` | Complete vector of finite numeric values; the first six positions are total sugar, reducing sugar, nicotine, chlorine, potassium, and total nitrogen, followed by three ratio fields | The five targets use indices `[0, 1, 2, 5, 4]` |
| `weather` | Matrix of finite numeric values ordered by date from May 1 to September 30, with shape `153 × 7` | Climate input construction |

The source weather-column order is `Tmax, Tmin, Tavg, Rain, Pressure, Sunshine, Wind`. The W7 modeling order constructed by the code is `Tmax, Tmin, Tavg, Rain, Sunshine, Pressure, Wind`. O6 and W6 select their specified source columns; O5 is obtained by removing Tavg from O6. Sunshine denotes mean net sunshine intensity in this study; the numeric ranges in the configuration should not be interpreted as sunshine duration in hours.

## Record order and category encoding

The construction scripts retain the relative order of records passing validation when generating climate arrays, target arrays, and `sample_info.csv`. Sample information includes `sample_id`, `original_index`, `year`, `place`, `grade`, and `json_source`. Subsequent grade IDs and split indices refer to positions in the processed arrays.

For the main study dataset, `sample_id == original_index == processed row position`. Excluding records during construction can change this correspondence; K326 selection and some subsequent analyses require the relevant alignment to be preserved. Reproducing the paper's fixed record split requires the same original record order and the same set of valid records. Seed 42 and the 70/15/15 proportions alone do not recover the same record identities.

The main dataset's raw grade field forms 16 observed categories, including grade labels and grade–variety combination labels. Script 09 converts grades to strings, strips leading and trailing whitespace, handles missing labels explicitly, and assigns IDs starting at 0 in category-frequency order. It does not separate grade from variety, merge aliases, or apply an additional sorting rule to categories with equal frequencies.

The category ID space is defined from category frequencies across the complete processed dataset. Grade mean target means are fitted using training records only. These two steps use different statistical quantities. The main and more-rules datasets have different category ID orders and must use their respective arrays and mappings.

## Local runtime artifacts

Running the code in an authorized environment can generate `sample_info.csv`, X/y and complete chemical arrays, grade IDs/mappings, splits, standardized arrays, scalers, sample-level predictions, record-level OOF predictions, statistical details linkable to county-years, checkpoints, and training logs. These files may contain actual observations, record-order linkages, or statistics fitted from training data. Providing the generation code does not grant redistribution rights for these artifacts.

K326 grouped tables with county-year identifiers and record-level differences are written to `src/core/k326_grade_validation/runtime_private/`, which is excluded by `.gitignore`. The released `src/core/k326_grade_validation/summary/group_cv_oof_cluster_differences.csv` contains only 47 rows of `delta_mape_pp`.

`.gitignore` excludes identified restricted inputs and local runtime artifacts. A complete post-run working directory should remain within the authorized data environment; only reviewed and approved files should be released publicly.

## Construction scope

`05_build_from_original_json.py` expects inputs conforming to the structure above, including complete chemical vectors and `153 × 7` weather matrices. It is not a general-purpose cleaning tool for arbitrary missing or invalid records.

`12_build_more_rules_dataset.py` reads `more-rules.json` after filtering according to the study definition; it does not implement the complete upstream filtering from the main dataset to the more-rules dataset. Its outputs are written to `src/core/data/processed_more/`. Even if script 05 is instructed to search for `more-rules.json`, it still writes to the main processed directory. Use the dedicated scripts 12/13 for the more-rules workflow.

The full execution order is given in the [Reproducibility guide](docs/REPRODUCIBILITY.md).
