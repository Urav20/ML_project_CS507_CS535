# Earthquake Building Damage Prediction

**UE24CS352A – Machine Learning Mini-Project**

| Team Member | SRN |
|---|---|
| Name 1: Urav B S | SRN: PES1UG24CS507 |
| Name 2: Vilohith Sree Sai Reddy Daggolu | SRN: PES1UG24CS535 |

---

## 1. Problem Statement

After a large earthquake, engineers need to know which buildings are badly damaged, but inspecting every
building by hand or by structural calculation is slow. This project uses data from the April 2015 Nepal
(Gorkha) earthquake to **predict how badly a building was damaged from its physical and legal
characteristics** (age, number of floors, foundation type, construction materials, location, etc.).

The target is `damage_grade`:

| Grade | Meaning |
|---|---|
| 1 | Low damage |
| 2 | Medium damage |
| 3 | Almost complete destruction |

This is a **multiclass classification** problem. We train and compare three algorithms:

**K-Nearest Neighbours (KNN)**, **Random Forest** and a **Neural Network**, using micro-F1 as the main
metric (as in the reference report) together with macro-F1, weighted-F1 and confusion matrices.

## 2. Dataset

| File | Rows | Content |
|---|---:|---|
| `train_values.csv` | 260,601 | 38 building features + `building_id` |
| `train_labels.csv` | 260,601 | `building_id` + `damage_grade` |
| `test_values.csv` | 86,868 | Same features, **no labels** |

**Findings from `exploration.py`:** no missing values, no duplicate rows or IDs.

**The 38 features by type:**

| Type | Count | Columns |
|---|---:|---|
| Geographic codes | 3 | `geo_level_1_id`, `geo_level_2_id`, `geo_level_3_id` |
| Numerical | 5 | `count_floors_pre_eq`, `age`, `area_percentage`, `height_percentage`, `count_families` |
| Categorical | 8 | `land_surface_condition`, `foundation_type`, `roof_type`, `ground_floor_type`, `other_floor_type`, `position`, `plan_configuration`, `legal_ownership_status` |
| Binary (0/1) | 22 | 11 `has_superstructure_*` + 11 `has_secondary_use*` |

`building_id` is only an identifier and is never used as a feature.

**Class imbalance:** Grade 1 = 9.64%, Grade 2 = 56.89%, Grade 3 = 33.47%.

## 3. Approach in Brief

- **Split:** 80% train / 20% validation, **stratified** (keeps the class mix), seed 42.

- **Geographic codes** are labels, not measurements, so they are not scaled as plain numbers.

  KNN and Neural Network: `geo_level_1_id` is one-hot encoded; `geo_level_2_id` and `geo_level_3_id`
  (thousands of values) are frequency-encoded. Random Forest uses the raw integer codes.

- **Categorical features:** one-hot encoding.

- **Binary features:** kept as 0/1.

- **Numerical features:** standardised for KNN and Neural Network only (Random Forest does not need scaling).

- **No data leakage:** every encoder and scaler sits inside a scikit-learn `Pipeline`, so it is fitted only
  on training rows. The test data is touched only by `predict_test.py`.

- **Tuning:** random search with 3-fold cross-validation on a 40,000-row sample of the training split
  only, scored by micro-F1. The best settings are then used to retrain on the full training split.

  The validation set plays no part in choosing settings.

## 4. Project Structure

```text
ML_project/

├── dataset/                    # put the three CSV files here
├── src/
│   ├── config.py               # paths, seed, split size, column groups, tuning settings
│   ├── data_loading.py         # loads CSVs, makes the stratified train/validation split
│   ├── preprocessing.py        # encoders and scalers (one pipeline for KNN/NN, one for RF)
│   ├── experiment.py           # shared routine: tune -> train -> evaluate -> save
│   ├── evaluation_utils.py     # metrics, confusion-matrix plots, reports
│   ├── train_knn.py            # RUN: KNN
│   ├── train_random_forest.py  # RUN: Random Forest
│   ├── train_neural_network.py # RUN: Neural Network
│   ├── evaluate_models.py      # RUN: comparison table + combined confusion matrices
│   └── predict_test.py         # RUN: predictions for the test set
├── results/                    # all outputs are saved here
├── docs/summary.pdf            # 2-page summary
├── exploration.py              # initial dataset exploration
├── requirements.txt
└── README.md