import os
import numpy as np
import matplotlib.pyplot as plt

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
