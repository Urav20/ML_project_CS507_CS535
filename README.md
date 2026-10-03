# Earthquake Building Damage Prediction

**UE24CS352A – Machine Learning Mini-Project**

| Team Member | SRN |
|---|---|
| Name 1: Urav B S| SRN: PES1UG24CS507 |
| Name 2: Vilohith Sree Sai Reddy Daggolu| SRN: PES1UG24CS535 |

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
|---|---|---|
| `train_values.csv` | 260,601 | 38 building features + `building_id` |
| `train_labels.csv` | 260,601 | `building_id` + `damage_grade` |
| `test_values.csv` | 86,868 | Same features, **no labels** |

**Findings from `exploration.py`:** no missing values, no duplicate rows or IDs.

**The 38 features by type:**

| Type | Count | Columns |
|---|---|---|
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
- **No data leakage:** every encoder and scaler sits inside a scikit-learn `Pipeline`, so it learns only from
  training rows. The test data is touched only by `predict_test.py`.
- **Tuning:** random search with 3-fold cross-validation on a 40,000-row sample **of the training split
  only**, scored by micro-F1. The best settings are then used to retrain on the full training split.
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
```

## 5. Setup

Requires Python 3.9 or newer.

```bash
# 1. create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt
```

## 6. How to Run

Run every command from the **project root folder** (the one containing `src/`), in this order:

```bash
python3 -m src.train_knn
python3 -m src.train_random_forest
python3 -m src.train_neural_network
python3 -m src.evaluate_models
python3 -m src.predict_test
```

(Use `python` instead of `python3` on Windows.)

### What each program prints and saves

| Program | What it does | Terminal output and meaning | Files saved in `results/` |
|---|---|---|---|
| `train_knn.py`, `train_random_forest.py`, `train_neural_network.py` | Tunes the model, trains it on the training split, scores it on the **validation** split | **Best params:** the winning settings. **Val accuracy / micro-F1 / macro-F1 / weighted-F1:** scores on unseen validation buildings (micro-F1 equals accuracy here). **Confusion matrix:** rows = actual grade, columns = predicted grade. **Row-normalised matrix:** percent of each actual grade sent to each prediction. **Per-class scores:** precision, recall, F1 for each grade. | `metrics/<model>.json`, `confusion_matrices/<model>.png`, `reports/<model>.txt`, `models/<model>.joblib` |
| `evaluate_models.py` | Puts the three models side by side | A comparison table (accuracy, macro precision/recall, micro/macro/weighted F1, train accuracy, training and prediction time) and per-grade recall for each model | `model_comparison.csv`, `model_comparison.md`, `confusion_matrices_all.png` |
| `predict_test.py` | Predicts a damage grade for all 86,868 test buildings using the model with the best validation micro-F1 (or choose one with `--model "Random Forest"`) | The model used, the number of rows saved, and the share of predictions per grade | `predictions/submission.csv` with columns `building_id,damage_grade` |

**Notes on reading the output**
- *Validation* scores are on 20% of the labelled data the model never trained on. The test set has no labels,
  so a true test score cannot be computed locally.
- The **Train Acc** column in the comparison table shows overfitting: a big gap between train and validation
  accuracy means the model memorised the training data. (KNN's value is not meaningful because each training
  building is its own nearest neighbour.)

### Quick demo mode (finishes in a minute or two)

Tuning on the full setup takes a while, and KNN needs about 90 seconds just to predict the validation set.
For a fast demo, reduce the tuning work:

```bash
# macOS / Linux
TUNE_SAMPLE=5000 N_ITER=2 python3 -m src.train_random_forest

# Windows PowerShell
$env:TUNE_SAMPLE=5000; $env:N_ITER=2; python -m src.train_random_forest
```

## 7. Our Results (validation set, 52,121 buildings)

| Model | Accuracy | Precision (macro) | Recall (macro) | Micro-F1 | Macro-F1 | Weighted-F1 | Training time (s) | Prediction time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KNN | 0.7026 | 0.6757 | 0.6246 | 0.7026 | 0.6450 | 0.6977 | 0.31 | 89.04 |
| Random Forest | **0.7146** | **0.7151** | 0.6109 | **0.7146** | 0.6433 | **0.7019** | 5.24 | 0.19 |
| Neural Network | 0.7009 | 0.6720 | 0.6237 | 0.7009 | 0.6427 | 0.6953 | 13.99 | 0.07 |

**Recall per damage grade**

| Model | Grade 1 | Grade 2 | Grade 3 |
|---|---:|---:|---:|
| KNN | 0.470 | 0.801 | 0.603 |
| Random Forest | 0.436 | 0.876 | 0.521 |
| Neural Network | 0.481 | 0.808 | 0.582 |

**Key observations**
- Random Forest has the best micro-F1 (0.7146), but the three models are within about 1.4 points of each
  other, and their macro-F1 values are almost tied (about 0.643-0.645).
- Grade 2 is identified most reliably. Grade 1 is hardest, because it is the rarest class (about 10%).
- Errors are mostly between **neighbouring** grades. Grade 1 mixed up with Grade 3 is under 2%.
  The models lean towards predicting "medium" (Grade 2).
- Random Forest shows some overfitting (about 86% on training vs 71.5% on validation); the Neural Network
  shows almost none.
- The reference report's scores were KNN 0.6488, Neural Network 0.6502 and Random Forest 0.7150
  (the last on a hidden test set); our validation scores are comparable or better.
- Test predictions from Random Forest: 6.0% Grade 1, 70.9% Grade 2, 23.0% Grade 3.

### Micro-F1 vs macro-F1

- **F1** for one grade blends precision (when the model says this grade, how often it is right) and recall
  (how many real buildings of this grade it finds).
- **Micro-F1** pools all predictions and scores once. For single-label multiclass data it equals accuracy,
  and the large Grade 2 class dominates it. It is this project's required metric.
- **Macro-F1** computes F1 per grade and averages the three equally, so weak performance on the rare
  Grade 1 pulls it down.
- **Weighted-F1** averages the per-grade F1 in proportion to class size.

Comparing micro and macro shows whether a model is only good on the common class.

## 8. Experiments: Changing the Code to Test for Better Results

Change **one thing at a time**, save the results of each run in its own folder so nothing is overwritten,
and revert the change before the next experiment:

```bash
RESULTS_DIR=results_exp1 python3 -m src.train_random_forest
```

Compare against the baseline above. Differences under about 0.5% may be noise.

| # | File | Change | Why it might help |
|---|---|---|---|
| 1 | `src/experiment.py` | `scoring="f1_micro"` to `scoring="f1_macro"` | The search then favours settings that do well on all three grades, which should raise Grade 1 recall, possibly at a small cost in accuracy |
| 2 | `src/config.py` | `TUNE_SAMPLE` from `40000` to `100000` | Settings are chosen on more data, so they suit the full training set better. Slower. |
| 3 | `src/config.py` | `N_ITER` from `8` to `20` | More setting combinations are tried, so a better one is more likely to be found |
| 4 | `src/config.py` | `CV_FOLDS` from `3` to `5` | Each candidate's score is more reliable, so the winner is less likely to be a fluke |
| 5a | `src/train_knn.py` | `n_neighbors` list to `[15, 31, 51, 101]` | The previous winner (31) was the largest value offered, so a higher value may be better |
| 5b | `src/train_random_forest.py` | `min_samples_leaf` to `[1, 2, 5, 10]` and add `"model__max_features": ["sqrt", 0.3]` | Larger leaves reduce memorising and may narrow the train/validation gap |
| 5c | `src/train_neural_network.py` | `hidden_layer_sizes` to `[(64,), (64, 32), (128, 64), (128, 64, 32)]` | The network is not overfitting, so a larger one may learn more |
| 6 | `src/config.py` | `SEED` from `42` to `7` (rerun all three models) | Shows whether the results and the model ranking are stable or just luck |
| 7 | `src/config.py` | `VAL_SIZE` from `0.20` to `0.30` | A larger validation set gives a steadier score, at the cost of less training data |
| 8 | `src/preprocessing.py` | Remove the geographic columns (delete the `geo_freq` block and `"geo_level_1_id"` for KNN/NN; remove `config.GEO_COLS` for Random Forest) | Shows how much location matters. A drop in score means location carries real information. |
| 9 | `src/preprocessing.py` | One-hot encode `geo_level_2_id` for Random Forest | Tests whether one-hot beats raw integer codes for trees. Much slower. |
| 10 | `src/config.py` | In `BIN_COLS`, remove the rare `has_secondary_use_*` columns (keep `has_secondary_use`) | Fewer, near-empty columns may reduce noise in KNN distances |
| 11 | `src/train_random_forest.py` | `"model__class_weight": [None, "balanced"]` to `["balanced"]` | Rare Grade 1 buildings count for more in training: expect higher Grade 1 recall, lower Grade 1 precision |
| 12 | Experiments 1 + 11 together | Both changes | Pushes Random Forest towards balanced performance across grades |

**What `TUNE_SAMPLE` and the grids do:** tuning trains each candidate several times, so it uses a
40,000-row sample of the training split to stay fast. The final model is retrained on all training rows.
The grid in each `train_*.py` file is the menu of settings the search may choose from, so it can only pick
values you list. If a winner sits at the edge of its menu, widening the menu can reveal a better value.

## 9. Limitations

- Random search tried only 8 setting combinations per model (for example 8 of 48 for Random Forest),
  so better settings may exist.
- Tuning used a 40,000-row sample and micro-F1, which favours the large Grade 2 class.
- Results come from one train/validation split; differences under about 1% could change with another split.
- Test labels are not provided, so test performance cannot be measured locally. Validation micro-F1
  (about 0.715 for Random Forest) is our estimate of real-world performance.
- Geographic codes with thousands of values are only encoded in simple ways (raw integer or frequency).

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `No module named src` or import errors | Run commands from the project root using `python3 -m src.train_knn`, not `python3 src/train_knn.py` |
| `FileNotFoundError ... train_values.csv` | Put the three CSV files in `dataset/` (or set the `DATA_DIR` environment variable) |
| `No module named sklearn` or `pandas` | Activate the virtual environment, then run `pip install -r requirements.txt` |
| `predict_test` or `evaluate_models` cannot find files | Train all three models first so `results/metrics/` and `results/models/` exist |
| Training is very slow | Use quick demo mode (Section 6); KNN prediction is slow by design |
| `python3` not found (Windows) | Use `python` instead |
