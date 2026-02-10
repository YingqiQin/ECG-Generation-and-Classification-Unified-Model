import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def per_class_ovr_metrics(y_true, y_pred, n_class: int):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_class)))  # [C,C]
    N = cm.sum()

    rows = []
    for k in range(n_class):
        TP = cm[k, k]
        FN = cm[k, :].sum() - TP
        FP = cm[:, k].sum() - TP
        TN = N - TP - FN - FP

        precision = TP / (TP + FP + 1e-12)
        recall = TP / (TP + FN + 1e-12)              # sensitivity
        specificity = TN / (TN + FP + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        acc_ovr = (TP + TN) / (N + 1e-12)
        bal_acc = 0.5 * (recall + specificity)

        rows.append({
            "class": k,
            "support": int(cm[k, :].sum()),
            "TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN),
            "precision_ovr": float(precision),
            "recall_sens_ovr": float(recall),
            "specificity_ovr": float(specificity),
            "f1_ovr": float(f1),
            "acc_ovr": float(acc_ovr),
            "balanced_acc_ovr": float(bal_acc),
        })

    df = pd.DataFrame(rows)

    # macro/weighted summaries (for these OVR metrics)
    macro = df[["precision_ovr","recall_sens_ovr","specificity_ovr","f1_ovr","balanced_acc_ovr"]].mean()
    w = df["support"].values / df["support"].sum()
    weighted = (df[["precision_ovr","recall_sens_ovr","specificity_ovr","f1_ovr","balanced_acc_ovr"]].values * w[:,None]).sum(axis=0)

    summary = pd.DataFrame({
        "avg": ["macro", "weighted"],
        "precision_ovr": [macro["precision_ovr"], weighted[0]],
        "recall_sens_ovr": [macro["recall_sens_ovr"], weighted[1]],
        "specificity_ovr": [macro["specificity_ovr"], weighted[2]],
        "f1_ovr": [macro["f1_ovr"], weighted[3]],
        "balanced_acc_ovr": [macro["balanced_acc_ovr"], weighted[4]],
    })

    return cm, df, summary

# 额外：常规多分类报告（非 OVR，直接来自 confusion matrix 的 per-class P/R/F1）
def standard_report(y_true, y_pred, target_names=None):
    return classification_report(y_true, y_pred, target_names=target_names, digits=4)
