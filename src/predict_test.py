"""Generate predictions for the unlabelled test set (run ONLY after model selection).
Usage: python -m src.predict_test [--model "Random Forest"]   (default: best val micro-F1)"""
import argparse
import json
import joblib
import pandas as pd
from src import config
from src.data_loading import load_test
from src.evaluate_models import NAMES


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=None)
    name = ap.parse_args().model
    if name is None:
        scores = {n: json.load(open(config.RESULTS_DIR / "metrics" / f"{n}.json"))["f1_micro"]
                  for n in NAMES}
        name = max(scores, key=scores.get)
    print("Using model:", name)
    pipe = joblib.load(config.RESULTS_DIR / "models" / f"{name}.joblib")
    X_test = load_test()
    sub = pd.DataFrame({config.ID_COL: X_test.index,
                        config.TARGET_COL: pipe.predict(X_test)})
    out_dir = config.RESULTS_DIR / "predictions"; out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "submission.csv"
    sub.to_csv(path, index=False)
    print(f"Saved {len(sub)} rows to {path}")
    print(sub[config.TARGET_COL].value_counts(normalize=True).sort_index().round(4))


if __name__ == "__main__":
    main()
