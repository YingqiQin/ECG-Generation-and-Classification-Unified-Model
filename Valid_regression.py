import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


def _safe_pearson(x, y):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return pearsonr(x, y)[0]

def _safe_spearman(x, y):
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan
    return spearmanr(x, y)[0]


def compute_lvef_regression_report(
    y_true,
    y_pred,
    out_dir: str = "lvef_report",
    prefix: str = "subject",
    thresholds=(40, 50),
    bins=((None, 35), (35, 50), (50, 60), (60, None)),
):
    """
    y_true, y_pred: array-like, per-subject LVEF (percentage points)
    out_dir: folder to save figures & csv
    prefix: file name prefix
    thresholds: classification thresholds to derive binary metrics: y=1 if LVEF < thr (you can flip below)
    bins: tuple of (low, high) for bucketing analysis on y_true
    """
    os.makedirs(out_dir, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = len(y_true)
    if n == 0:
        raise ValueError("No valid samples after removing NaN/Inf.")

    err = y_pred - y_true
    abs_err = np.abs(err)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = float(np.mean(err))
    sd = float(np.std(err, ddof=1)) if n > 1 else np.nan
    r2 = r2_score(y_true, y_pred) if n > 1 else np.nan
    pcc = _safe_pearson(y_true, y_pred)
    spc = _safe_spearman(y_true, y_pred)

    p_le_5 = float(np.mean(abs_err <= 5.0))
    p_le_10 = float(np.mean(abs_err <= 10.0))
    p_le_3 = float(np.mean(abs_err <= 3.0))

    # Bland–Altman
    mean_pair = (y_true + y_pred) / 2.0
    loa_low = bias - 1.96 * sd if np.isfinite(sd) else np.nan
    loa_high = bias + 1.96 * sd if np.isfinite(sd) else np.nan

    # Save summary json-like dict and csv
    summary = {
        "n": int(n),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias_mean_error": float(bias),
        "sd_error": float(sd),
        "loa_low": float(loa_low) if np.isfinite(loa_low) else np.nan,
        "loa_high": float(loa_high) if np.isfinite(loa_high) else np.nan,
        "pearson_r": float(pcc) if np.isfinite(pcc) else np.nan,
        "spearman_r": float(spc) if np.isfinite(spc) else np.nan,
        "r2": float(r2) if np.isfinite(r2) else np.nan,
        "p_abs_err_le_3": float(p_le_3),
        "p_abs_err_le_5": float(p_le_5),
        "p_abs_err_le_10": float(p_le_10),
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(out_dir, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # Per-sample table (optional, helpful for analysis)
    per_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "error": err,
        "abs_error": abs_err,
        "mean_pair": mean_pair
    })
    per_path = os.path.join(out_dir, f"{prefix}_per_sample.csv")
    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")

    # ---------- Bucket analysis ----------
    bucket_rows = []
    for low, high in bins:
        if low is None:
            idx = y_true < high
            name = f"<{high}"
        elif high is None:
            idx = y_true >= low
            name = f">={low}"
        else:
            idx = (y_true >= low) & (y_true < high)
            name = f"[{low},{high})"

        if idx.sum() == 0:
            bucket_rows.append({
                "bucket": name, "n": 0,
                "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "p(|e|<=5)": np.nan, "p(|e|<=10)": np.nan
            })
            continue

        yt = y_true[idx]; yp = y_pred[idx]
        e = yp - yt; ae = np.abs(e)
        bucket_rows.append({
            "bucket": name, "n": int(idx.sum()),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "bias": float(np.mean(e)),
            "p(|e|<=5)": float(np.mean(ae <= 5.0)),
            "p(|e|<=10)": float(np.mean(ae <= 10.0)),
        })

    bucket_df = pd.DataFrame(bucket_rows)
    bucket_path = os.path.join(out_dir, f"{prefix}_bucket_report.csv")
    bucket_df.to_csv(bucket_path, index=False, encoding="utf-8-sig")

    # ---------- Figures ----------
    # 1) Scatter y_true vs y_pred
    fig = plt.figure()
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("LVEF True")
    plt.ylabel("LVEF Pred")
    plt.title(
        f"{prefix}: y_true vs y_pred | MAE={mae:.2f}, RMSE={rmse:.2f}, r={pcc:.2f}, R2={r2:.2f}"
    )
    fig.savefig(os.path.join(out_dir, f"{prefix}_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2) Error histogram
    fig = plt.figure()
    plt.hist(err, bins=50)
    plt.xlabel("Error (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"{prefix}: Error distribution | bias={bias:.2f}, sd={sd:.2f}")
    fig.savefig(os.path.join(out_dir, f"{prefix}_error_hist.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3) Abs error histogram
    fig = plt.figure()
    plt.hist(abs_err, bins=50)
    plt.xlabel("|Error|")
    plt.ylabel("Count")
    plt.title(f"{prefix}: |Error| distribution | P(|e|<=5)={p_le_5:.3f}, P(|e|<=10)={p_le_10:.3f}")
    fig.savefig(os.path.join(out_dir, f"{prefix}_abserr_hist.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4) Bland–Altman plot
    fig = plt.figure()
    plt.scatter(mean_pair, err, s=12, alpha=0.6)
    plt.axhline(bias, linestyle="--")
    if np.isfinite(loa_low) and np.isfinite(loa_high):
        plt.axhline(loa_low, linestyle="--")
        plt.axhline(loa_high, linestyle="--")
        txt = f"bias={bias:.2f}\nLoA=[{loa_low:.2f}, {loa_high:.2f}]"
    else:
        txt = f"bias={bias:.2f}"
    plt.xlabel("Mean of (True, Pred)")
    plt.ylabel("Error (Pred - True)")
    plt.title(f"{prefix}: Bland–Altman\n{txt}")
    fig.savefig(os.path.join(out_dir, f"{prefix}_bland_altman.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5) Residual vs y_true (heteroscedasticity)
    fig = plt.figure()
    plt.scatter(y_true, err, s=12, alpha=0.6)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("LVEF True")
    plt.ylabel("Error (Pred - True)")
    plt.title(f"{prefix}: Residual vs True")
    fig.savefig(os.path.join(out_dir, f"{prefix}_residual_vs_true.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- Derived classification metrics ----------
    # By default: positive = (LVEF < thr)
    cls_rows = []
    for thr in thresholds:
        y_true_cls = (y_true < thr).astype(int)
        # derive prob of positive via regression: map to a score; simplest is use -y_pred (monotonic)
        # Here we use score = -y_pred so lower predicted LVEF => higher probability of positive (LVEF<thr).
        # This is for AUROC/AUPRC ranking only.
        score = -y_pred
        # Normalize score to [0,1] (monotonic transform doesn't change AUROC)
        score01 = (score - score.min()) / (score.max() - score.min() + 1e-12)

        # hard prediction at same threshold
        y_pred_cls = (y_pred < thr).astype(int)
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])

        # AUROC/AUPRC for this derived classifier
        if len(np.unique(y_true_cls)) > 1:
            auroc = roc_auc_score(y_true_cls, score01)
            auprc = average_precision_score(y_true_cls, score01)
        else:
            auroc = np.nan
            auprc = np.nan

        cls_rows.append({
            "threshold": thr,
            "pos_def": f"LVEF < {thr}",
            "pos_ratio": float(y_true_cls.mean()),
            "auroc_rank_from_reg": float(auroc) if auroc == auroc else np.nan,
            "auprc_rank_from_reg": float(auprc) if auprc == auprc else np.nan,
            "cm_00_tn": int(cm[0, 0]), "cm_01_fp": int(cm[0, 1]),
            "cm_10_fn": int(cm[1, 0]), "cm_11_tp": int(cm[1, 1]),
        })

    cls_df = pd.DataFrame(cls_rows)
    cls_path = os.path.join(out_dir, f"{prefix}_derived_classification.csv")
    cls_df.to_csv(cls_path, index=False, encoding="utf-8-sig")

    print("Saved report to:", out_dir)
    print("Summary CSV:", summary_path)
    print("Bucket CSV:", bucket_path)
    print("Derived CLS CSV:", cls_path)
    print("Per-sample CSV:", per_path)

    return {
        "summary": summary,
        "summary_csv": summary_path,
        "bucket_csv": bucket_path,
        "derived_cls_csv": cls_path,
        "per_sample_csv": per_path,
        "out_dir": out_dir,
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Replace with your arrays
    y_true = [55, 45, 60, 35]
    y_pred = [52, 50, 58, 40]

    compute_lvef_regression_report(
        y_true, y_pred,
        out_dir="lvef_report",
        prefix="ppg_lvef"
    )
