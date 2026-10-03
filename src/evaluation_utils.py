"""Metrics, confusion-matrix plot, classification report."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             precision_recall_fscore_support)
from src import config

LABELS = ["1 (Low)", "2 (Medium)", "3 (Destroyed)"]


def evaluate(y_true, y_pred):
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ["micro", "macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, average=avg, zero_division=0)
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = p, r, f
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=config.CLASSES, zero_division=0)
    out["per_class"] = {str(c): {"precision": p[i], "recall": r[i], "f1": f[i],
                                 "support": int(s[i])}
                        for i, c in enumerate(config.CLASSES)}
    out["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=config.CLASSES).tolist()
    return out


def draw_confusion(ax, cm, title):
    cm = np.array(cm)
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.0%})", ha="center", va="center",
                    color="white" if pct[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(LABELS, fontsize=8); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)


def save_confusion_matrix(cm, name, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    draw_confusion(ax, cm, name)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_report(y_true, y_pred, path):
    with open(path, "w") as f:
        f.write(classification_report(y_true, y_pred, labels=config.CLASSES, digits=4,
                                      zero_division=0))
