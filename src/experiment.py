"""Shared training routine used by all three models, so the setup is identical:
same split -> tune on a TRAIN-only subsample with CV -> refit on full train
-> evaluate once on the validation set -> save results."""
import json
import time
from math import prod
import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from src import config
from src.data_loading import load_train, split_train_val
from src.evaluation_utils import evaluate, save_confusion_matrix, save_report


def run_experiment(name, preprocessor, estimator, param_grid):
    for sub in ["metrics", "confusion_matrices", "reports", "models"]:
        (config.RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

    X, y = load_train()
    X_tr, X_val, y_tr, y_val = split_train_val(X, y)
    print(f"[{name}] train={len(X_tr)} validation={len(X_val)}")

    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])

    # ---- 1. hyperparameter tuning (uses ONLY training rows) ----
    sample = X_tr.sample(n=min(config.TUNE_SAMPLE, len(X_tr)), random_state=config.SEED)
    n_iter = min(config.N_ITER, prod(len(v) for v in param_grid.values()))
    search = RandomizedSearchCV(
        pipe, param_grid, n_iter=n_iter, scoring="f1_micro",
        cv=StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.SEED),
        refit=False, random_state=config.SEED, verbose=1)
    search.fit(sample, y_tr.loc[sample.index])
    best = search.best_params_
    print(f"[{name}] best params: {best} (CV micro-F1 {search.best_score_:.4f})")

    # ---- 2. final fit on the full training split ----
    pipe.set_params(**best)
    t0 = time.time(); pipe.fit(X_tr, y_tr); train_time = time.time() - t0

    # ---- 3. evaluate on validation set ----
    t0 = time.time(); pred = pipe.predict(X_val); predict_time = time.time() - t0
    result = evaluate(y_val, pred)
    result.update(model=name, best_params={k: str(v) for k, v in best.items()},
                  cv_micro_f1_on_tuning_sample=search.best_score_,
                  train_time_sec=train_time, predict_time_sec=predict_time)
    # train-set score on 20k rows -> shows over-fitting (train vs validation gap)
    result["train_accuracy"] = float((pipe.predict(X_tr.iloc[:20000]) ==
                                      y_tr.iloc[:20000]).mean())

    # ---- 4. save ----
    r = config.RESULTS_DIR
    json.dump(result, open(r / "metrics" / f"{name}.json", "w"), indent=2)
    save_confusion_matrix(result["confusion_matrix"], name,
                          r / "confusion_matrices" / f"{name}.png")
    save_report(y_val, pred, r / "reports" / f"{name}.txt")
    joblib.dump(pipe, r / "models" / f"{name}.joblib", compress=3)
    print(f"[{name}] val accuracy={result['accuracy']:.4f} micro-F1={result['f1_micro']:.4f} "
          f"macro-F1={result['f1_macro']:.4f} weighted-F1={result['f1_weighted']:.4f} "
          f"train={train_time:.0f}s")
    print_confusion(result)
    return result


def print_confusion(result):
    """Print the confusion matrix (rows = actual, columns = predicted) and per-class scores."""
    names = ["Grade 1 (Low)", "Grade 2 (Medium)", "Grade 3 (Destroyed)"]
    cm = pd.DataFrame(result["confusion_matrix"],
                      index=[f"Actual {n}" for n in names],
                      columns=[f"Pred {i}" for i in (1, 2, 3)])
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm.to_string())
    print("\nRow-normalised (what % of each actual grade went where):")
    print((cm.div(cm.sum(axis=1), axis=0) * 100).round(1).to_string())
    pc = pd.DataFrame(result["per_class"]).T[["precision", "recall", "f1", "support"]]
    pc.index = names
    print("\nPer-class scores:")
    print(pc.round(3).to_string())