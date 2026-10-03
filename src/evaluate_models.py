"""Combine the three saved metric files into one table + one confusion figure."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src import config
from src.evaluation_utils import draw_confusion

NAMES = ["KNN", "Random Forest", "Neural Network"]


def main():
    res = {n: json.load(open(config.RESULTS_DIR / "metrics" / f"{n}.json")) for n in NAMES}
    df = pd.DataFrame([{
        "Model": n, "Accuracy": r["accuracy"],
        "Precision (macro)": r["precision_macro"], "Recall (macro)": r["recall_macro"],
        "Micro-F1": r["f1_micro"], "Macro-F1": r["f1_macro"],
        "Weighted-F1": r["f1_weighted"], "Train Acc (20k rows)": r["train_accuracy"],
        "Training Time (s)": r["train_time_sec"], "Prediction Time (s)": r["predict_time_sec"],
    } for n, r in res.items()])
    df.round(4).to_csv(config.RESULTS_DIR / "model_comparison.csv", index=False)
    cols = list(df.columns)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        md.append("| " + " | ".join(str(row[c]) if c == "Model" else f"{row[c]:.4f}"
                                    for c in cols) + " |")
    (config.RESULTS_DIR / "model_comparison.md").write_text("\n".join(md))
    print("\n".join(md))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, n in zip(axes, NAMES):
        draw_confusion(ax, res[n]["confusion_matrix"], n)
    fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "confusion_matrices_all.png", dpi=150)

    print("\nPer-class recall:")
    for n in NAMES:
        print(n, {k: round(v["recall"], 3) for k, v in res[n]["per_class"].items()})


if __name__ == "__main__":
    main()
