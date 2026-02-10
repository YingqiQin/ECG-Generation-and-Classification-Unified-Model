import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------
# Config
# -----------------------------
FILES = {
    "stats": "train_oof_stats.csv",
    "flagged": "train_oof_stats_flagged.csv",
    "drop": "train_drop_list.csv",               # 你说的文件名
    "view01": "train_view_01_highconf.csv",      # 你说的文件名
}

OUT_DIR = "figures"
REPORT_PDF = "datapicking_report.pdf"

# If you want to show before/after AUC in a plot, fill these numbers.
AUC_BEFORE = {"macro": 0.74, "ovr": [0.734, 0.649, 0.732, 0.844]}
AUC_AFTER  = {"macro": 0.78, "ovr": None}  # optionally fill your new per-class OVR AUC here


def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _infer_flag_cols(df: pd.DataFrame):
    # expected columns from our earlier template
    candidates = ["flag_A1", "flag_B1", "flag_C1", "flag_01_conflict", "suggest_drop"]
    return [c for c in candidates if c in df.columns]


def _basic_summary(flagged: pd.DataFrame):
    # summary table: counts & rates per class
    cols = ["y"]
    for c in ["sample_id", "patient_id", "pred", "loss", "p_true", "p_max", "margin", "entropy"]:
        if c in flagged.columns:
            cols.append(c)

    df = flagged.copy()
    if "suggest_drop" not in df.columns:
        # fall back: treat any flag as drop
        flag_cols = _infer_flag_cols(df)
        if flag_cols:
            df["suggest_drop"] = df[flag_cols].any(axis=1)
        else:
            df["suggest_drop"] = False

    group = df.groupby("y", dropna=False)
    summary = pd.DataFrame({
        "n_total": group.size(),
        "n_drop": group["suggest_drop"].sum(),
        "drop_rate": group["suggest_drop"].mean(),
    })

    # optional: rule counts
    for rule in ["flag_A1", "flag_B1", "flag_C1", "flag_01_conflict"]:
        if rule in df.columns:
            summary[f"n_{rule}"] = group[rule].sum()
            summary[f"rate_{rule}"] = group[rule].mean()

    summary = summary.reset_index()
    return df, summary


def plot_waterfall_and_byclass(summary: pd.DataFrame, flagged: pd.DataFrame, savepath: str):
    # 1) Overall pipeline bar (approximate)
    total = len(flagged)
    n_drop = int(flagged["suggest_drop"].sum()) if "suggest_drop" in flagged.columns else 0
    n_keep = total - n_drop

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(["Total", "Dropped", "Kept"], [total, n_drop, n_keep])
    ax1.set_title("Data picking summary (train)")
    ax1.set_ylabel("Count")

    # 2) per-class drop rate
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.bar(summary["y"].astype(str), summary["drop_rate"])
    ax2.set_title("Drop rate by class")
    ax2.set_xlabel("Class (y)")
    ax2.set_ylabel("Drop rate")
    ax2.set_ylim(0, max(0.05, summary["drop_rate"].max() * 1.2))

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    return fig


def plot_loss_distributions(flagged: pd.DataFrame, savepath: str):
    # one subplot per class (but no subplots requested? It's okay to have multiple axes in one figure,
    # but you earlier didn't restrict. If you want strictly one plot per figure, split them.
    classes = sorted(flagged["y"].unique())
    n = len(classes)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, classes):
        sub = flagged[flagged["y"] == c]
        ax.hist(sub["loss"].values, bins=60)
        ax.set_title(f"Loss distribution (class {c})")

        # threshold line: if flag_B1 exists, approximate threshold as min(loss) among flagged_B1
        if "flag_B1" in sub.columns and sub["flag_B1"].any():
            thr = sub.loc[sub["flag_B1"], "loss"].min()
            ax.axvline(thr, linewidth=2)
            ax.text(thr, ax.get_ylim()[1] * 0.9, "B1 thr", rotation=90, va="top")

    axes[-1].set_xlabel("Cross-entropy loss (OOF)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    return fig


def plot_ptrue_vs_loss(flagged: pd.DataFrame, savepath: str):
    df = flagged.copy()
    if "suggest_drop" not in df.columns:
        df["suggest_drop"] = False

    kept = df[~df["suggest_drop"]]
    dropped = df[df["suggest_drop"]]

    fig = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    ax.scatter(kept["p_true"], kept["loss"], s=8, alpha=0.4, marker=".")
    ax.scatter(dropped["p_true"], dropped["loss"], s=18, alpha=0.8, marker="x")
    ax.set_xlabel("p_true")
    ax.set_ylabel("loss")
    ax.set_title("p_true vs loss (kept vs dropped)")

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    return fig


def _confusion_matrix(y_true, y_pred, n_class=4):
    cm = np.zeros((n_class, n_class), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_class and 0 <= p < n_class:
            cm[t, p] += 1
    return cm


def plot_confusion_triptych(flagged: pd.DataFrame, savepath: str, n_class=4):
    df = flagged.copy()
    if "suggest_drop" not in df.columns:
        df["suggest_drop"] = False

    all_cm = _confusion_matrix(df["y"].values, df["pred"].values, n_class=n_class)
    drop_cm = _confusion_matrix(df.loc[df["suggest_drop"], "y"].values,
                                df.loc[df["suggest_drop"], "pred"].values, n_class=n_class)
    keep_cm = _confusion_matrix(df.loc[~df["suggest_drop"], "y"].values,
                                df.loc[~df["suggest_drop"], "pred"].values, n_class=n_class)

    fig = plt.figure(figsize=(12, 4))
    for i, (cm, title) in enumerate([(all_cm, "All"), (drop_cm, "Dropped subset"), (keep_cm, "Kept subset")], start=1):
        ax = fig.add_subplot(1, 3, i)
        im = ax.imshow(cm)
        ax.set_title(title)
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_xticks(range(n_class))
        ax.set_yticks(range(n_class))
        # annotate counts
        for r in range(n_class):
            for c in range(n_class):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    return fig


def plot_view01(view01: pd.DataFrame, savepath_prefix: str):
    figs = []
    if len(view01) == 0:
        return figs

    # (y,pred) counts
    fig1 = plt.figure(figsize=(6, 4))
    ax = plt.gca()
    ct = view01.groupby(["y", "pred"]).size().reset_index(name="count")
    labels = [f"{int(r.y)}→{int(r.pred)}" for r in ct.itertuples(index=False)]
    ax.bar(labels, ct["count"].values)
    ax.set_title("0↔1 high-conf conflicts: (y→pred) counts")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig1.tight_layout()
    fig1.savefig(savepath_prefix + "_counts.png", dpi=200)
    figs.append(fig1)

    # p_max histogram
    if "p_max" in view01.columns:
        fig2 = plt.figure(figsize=(6, 4))
        plt.hist(view01["p_max"].values, bins=40)
        plt.title("p_max distribution (high-conf list)")
        plt.xlabel("p_max")
        plt.ylabel("Count")
        fig2.tight_layout()
        fig2.savefig(savepath_prefix + "_pmax.png", dpi=200)
        figs.append(fig2)

    # margin histogram
    if "margin" in view01.columns:
        fig3 = plt.figure(figsize=(6, 4))
        plt.hist(view01["margin"].values, bins=40)
        plt.title("margin distribution (high-conf list)")
        plt.xlabel("margin")
        plt.ylabel("Count")
        fig3.tight_layout()
        fig3.savefig(savepath_prefix + "_margin.png", dpi=200)
        figs.append(fig3)

    return figs


def plot_patient_effect(flagged: pd.DataFrame, savepath: str):
    if "patient_id" not in flagged.columns or flagged["patient_id"].isna().all():
        return None

    df = flagged.copy()
    if "suggest_drop" not in df.columns:
        df["suggest_drop"] = False

    g = df.groupby("patient_id")
    n_rec = g.size()
    drop_rate = g["suggest_drop"].mean()

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(n_rec.values, bins=50)
    ax1.set_title("Records per patient")
    ax1.set_xlabel("#ECGs")
    ax1.set_ylabel("Patients")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(drop_rate.values, bins=50)
    ax2.set_title("Drop rate per patient")
    ax2.set_xlabel("Drop rate")
    ax2.set_ylabel("Patients")

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    return fig


def plot_auc_before_after(savepath: str):
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()

    bars = [AUC_BEFORE["macro"], AUC_AFTER["macro"]]
    ax.bar(["Before", "After"], bars)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Macro AUC: before vs after data picking")
    ax.set_ylabel("Macro AUC")

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    return fig


def main():
    _ensure_dir(OUT_DIR)

    flagged = _safe_read_csv(FILES["flagged"])
    stats = None
    if os.path.exists(FILES["stats"]):
        stats = _safe_read_csv(FILES["stats"])

    drop_df = _safe_read_csv(FILES["drop"]) if os.path.exists(FILES["drop"]) else None
    view01 = _safe_read_csv(FILES["view01"]) if os.path.exists(FILES["view01"]) else pd.DataFrame()

    flagged, summary = _basic_summary(flagged)
    summary.to_csv(os.path.join(OUT_DIR, "summary_by_class.csv"), index=False)

    figs = []

    figs.append(plot_waterfall_and_byclass(summary, flagged, os.path.join(OUT_DIR, "fig1_pipeline.png")))
    figs.append(plot_loss_distributions(flagged, os.path.join(OUT_DIR, "fig2_loss_by_class.png")))
    figs.append(plot_ptrue_vs_loss(flagged, os.path.join(OUT_DIR, "fig3_ptrue_vs_loss.png")))
    figs.append(plot_confusion_triptych(flagged, os.path.join(OUT_DIR, "fig4_confusion_triptych.png")))

    view01_figs = plot_view01(view01, os.path.join(OUT_DIR, "fig5_view01"))
    figs.extend(view01_figs)

    fig_pat = plot_patient_effect(flagged, os.path.join(OUT_DIR, "fig6_patient_effect.png"))
    if fig_pat is not None:
        figs.append(fig_pat)

    figs.append(plot_auc_before_after(os.path.join(OUT_DIR, "fig7_macro_auc_before_after.png")))

    # Multi-page PDF
    with PdfPages(os.path.join(OUT_DIR, REPORT_PDF)) as pdf:
        # Add a front-page text summary
        fig0 = plt.figure(figsize=(11, 8.5))
        plt.axis("off")
        txt = []
        txt.append("Data picking / Loss-based cleaning report")
        txt.append("")
        txt.append(f"Total train samples: {len(flagged)}")
        txt.append(f"Dropped: {int(flagged['suggest_drop'].sum())} ({flagged['suggest_drop'].mean()*100:.2f}%)")
        txt.append("")
        txt.append("Per-class summary:")
        txt.append(summary.to_string(index=False))
        plt.text(0.02, 0.98, "\n".join(txt), va="top", family="monospace")
        pdf.savefig(fig0)
        plt.close(fig0)

        for fig in figs:
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[OK] Saved figures to: {OUT_DIR}/")
    print(f"[OK] Saved PDF report to: {OUT_DIR}/{REPORT_PDF}")
    print(f"[OK] Saved summary table to: {OUT_DIR}/summary_by_class.csv")


if __name__ == "__main__":
    main()
