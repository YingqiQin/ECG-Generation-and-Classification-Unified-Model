import os
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, precision_recall_curve
)

def compute_binary_metrics_with_curves(y_true, y_prob, threshold=0.5):
    """
    y_true: (N,) int {0,1}
    y_prob: (N,) float in [0,1] for positive class
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Curves only defined if both classes exist
    if len(np.unique(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)

        fpr, tpr, roc_thr = roc_curve(y_true, y_prob)
        pr_prec, pr_rec, pr_thr = precision_recall_curve(y_true, y_prob)
    else:
        auroc = np.nan
        auprc = np.nan
        fpr = tpr = roc_thr = None
        pr_prec = pr_rec = pr_thr = None

    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "threshold": float(threshold),
        "confusion_matrix": cm,
        "n": int(len(y_true)),
        "curves": {
            "roc": {"fpr": fpr, "tpr": tpr, "thr": roc_thr},
            "pr": {"precision": pr_prec, "recall": pr_rec, "thr": pr_thr},
        }
    }

def plot_roc(metrics, title="ROC", save_path=None):
    roc = metrics["curves"]["roc"]
    if roc["fpr"] is None:
        return None

    fig = plt.figure()
    plt.plot(roc["fpr"], roc["tpr"])
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUROC={metrics['auroc']:.3f})")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pr(metrics, title="Precision-Recall", save_path=None):
    pr = metrics["curves"]["pr"]
    if pr["precision"] is None:
        return None

    fig = plt.figure()
    plt.plot(pr["recall"], pr["precision"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} (AUPRC={metrics['auprc']:.3f})")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_confusion_matrix(cm, class_names=("0", "1"), normalize=False, title="Confusion Matrix", save_path=None):
    """
    cm: 2x2 numpy array
    normalize: if True, rows sum to 1 (per-true-class normalization)
    """
    cm = np.asarray(cm).astype(float)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, np.maximum(row_sum, 1e-12))

    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title + (" (norm)" if normalize else ""))
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = format(val, fmt) if normalize else str(int(val))
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if val > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Pred")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_all_binary_figures(metrics, out_dir, tag):
    """
    Saves:
      - roc_{tag}.png
      - pr_{tag}.png
      - cm_{tag}.png
      - cm_norm_{tag}.png
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_roc(metrics, title=f"ROC {tag}", save_path=os.path.join(out_dir, f"roc_{tag}.png"))
    plot_pr(metrics, title=f"PR {tag}", save_path=os.path.join(out_dir, f"pr_{tag}.png"))

    cm = metrics["confusion_matrix"]
    plot_confusion_matrix(cm, class_names=("0", "1"),
                          normalize=False, title=f"CM {tag}",
                          save_path=os.path.join(out_dir, f"cm_{tag}.png"))
    plot_confusion_matrix(cm, class_names=("0", "1"),
                          normalize=True, title=f"CM {tag}",
                          save_path=os.path.join(out_dir, f"cm_norm_{tag}.png"))
