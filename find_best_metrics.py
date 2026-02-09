import numpy as np
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, precision_recall_curve
)

def _to_p_pos(all_prob_or_logits, pos_index=1, already_prob=True):
    x = np.asarray(all_prob_or_logits)
    if x.ndim == 1:
        return x.astype(float)
    assert x.shape[1] == 2
    if already_prob:
        return x[:, pos_index].astype(float)
    # logits -> softmax
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    prob = exp / exp.sum(axis=1, keepdims=True)
    return prob[:, pos_index].astype(float)

def metrics_at_threshold(y_true, p_pos, thr):
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)
    y_pred = (p_pos >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)   # sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp + 1e-12)
    bal_acc = 0.5 * (rec + spec)

    return {
        "thr": float(thr),
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "balanced_acc": float(bal_acc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cm": cm
    }

def find_best_threshold(y_true, p_pos, strategy="max_f1", grid=1001, target=0.90):
    """
    strategy:
      - "max_f1"
      - "max_balanced_acc"
      - "youden_j"           (maximize TPR - FPR)
      - "spec_at_least"      (max recall subject to specificity >= target)
      - "recall_at_least"    (max specificity subject to recall >= target)
    """
    y_true = np.asarray(y_true).astype(int)
    p_pos = np.asarray(p_pos).astype(float)

    thrs = np.linspace(0.0, 1.0, grid)
    stats = [metrics_at_threshold(y_true, p_pos, t) for t in thrs]

    if strategy == "max_f1":
        best = max(stats, key=lambda d: d["f1"])
    elif strategy == "max_balanced_acc":
        best = max(stats, key=lambda d: d["balanced_acc"])
    elif strategy == "youden_j":
        # J = TPR - FPR = recall - (1 - specificity) = recall + specificity - 1
        best = max(stats, key=lambda d: d["recall"] + d["specificity"] - 1.0)
    elif strategy == "spec_at_least":
        cand = [d for d in stats if d["specificity"] >= target]
        best = max(cand, key=lambda d: d["recall"]) if len(cand) else max(stats, key=lambda d: d["specificity"])
    elif strategy == "recall_at_least":
        cand = [d for d in stats if d["recall"] >= target]
        best = max(cand, key=lambda d: d["specificity"]) if len(cand) else max(stats, key=lambda d: d["recall"])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return best, stats

