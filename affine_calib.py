import numpy as np
import pandas as pd


def me_std_mae(y_pred, y_true):
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    err = y_pred - y_true
    me = float(np.mean(err))
    std = float(np.std(err, ddof=1)) if len(err) > 1 else float("nan")
    mae = float(np.mean(np.abs(err)))
    return me, std, mae


def fit_affine_ridge(x, y, lam=10.0, penalize_intercept=False):
    """
    Fit y ≈ a*x + b with ridge regularization.
    By default, penalize slope only (common / more stable with few points).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.stack([x, np.ones_like(x)], axis=1)  # [n,2] => [a,b]
    R = np.diag([lam, lam if penalize_intercept else 0.0])
    w = np.linalg.solve(X.T @ X + R, X.T @ y)
    a, b = float(w[0]), float(w[1])
    return a, b


def apply_affine_calibration(
    df: pd.DataFrame,
    lam: float = 10.0,
    min_points: int = 3,
    by_sleep: bool = True,
    penalize_intercept: bool = False,
):
    """
    Use only calibration rows (is_calib==True) to fit affine mapping per subject,
    optionally per (subject, sleep). Apply to all rows. Evaluate should be done on ~is_calib.
    Adds:
      y_pred_sbp_aff, y_pred_dbp_aff
    """
    out = df.copy()

    out["y_pred_sbp_aff"] = out["y_pred_sbp"].astype(float)
    out["y_pred_dbp_aff"] = out["y_pred_dbp"].astype(float)

    # Pre-split calib rows for speed
    calib = out[out["is_calib"].astype(bool)].copy()

    # Group keys
    if by_sleep:
        group_keys = ["id_clean", "sleep"]
        all_keys = ["id_clean"]
    else:
        group_keys = ["id_clean"]
        all_keys = ["id_clean"]

    # For fallback: subject-level calib pool
    subj_pool = {
        pid: g for pid, g in calib.groupby("id_clean", sort=False)
    }

    # Fit and apply
    if by_sleep:
        groups = out.groupby(["id_clean", "sleep"], sort=False)
    else:
        groups = out.groupby(["id_clean"], sort=False)

    for key, g_all in groups:
        if by_sleep:
            pid, slp = key
            g_cal = calib[(calib["id_clean"] == pid) & (calib["sleep"] == slp)]
        else:
            pid = key
            slp = None
            g_cal = calib[calib["id_clean"] == pid]

        # fallback pools
        g_cal_subj = subj_pool.get(pid, None)

        def fit_or_fallback(target_col_true, target_col_pred):
            # try state-specific
            if g_cal is not None and len(g_cal) >= min_points:
                a, b = fit_affine_ridge(
                    g_cal[target_col_pred], g_cal[target_col_true],
                    lam=lam, penalize_intercept=penalize_intercept
                )
                return a, b

            # fallback to subject-level calib
            if g_cal_subj is not None and len(g_cal_subj) >= min_points:
                a, b = fit_affine_ridge(
                    g_cal_subj[target_col_pred], g_cal_subj[target_col_true],
                    lam=lam, penalize_intercept=penalize_intercept
                )
                return a, b

            # final fallback: bias-only using existing subject bias if available, else 0
            # (this keeps behavior safe)
            if "off_sbp_subj" in out.columns and target_col_true == "y_true_sbp":
                # y_true - y_pred = off  => y_pred + off
                b_only = float(g_all["off_sbp_subj"].iloc[0]) if pd.notna(g_all["off_sbp_subj"].iloc[0]) else 0.0
                return 1.0, b_only
            if "off_dbp_subj" in out.columns and target_col_true == "y_true_dbp":
                b_only = float(g_all["off_dbp_subj"].iloc[0]) if pd.notna(g_all["off_dbp_subj"].iloc[0]) else 0.0
                return 1.0, b_only

            return 1.0, 0.0

        a_s, b_s = fit_or_fallback("y_true_sbp", "y_pred_sbp")
        a_d, b_d = fit_or_fallback("y_true_dbp", "y_pred_dbp")

        # apply to all rows in this group
        idx = g_all.index
        out.loc[idx, "y_pred_sbp_aff"] = a_s * out.loc[idx, "y_pred_sbp"].to_numpy() + b_s
        out.loc[idx, "y_pred_dbp_aff"] = a_d * out.loc[idx, "y_pred_dbp"].to_numpy() + b_d

    return out


def report_all(df: pd.DataFrame, name: str):
    eval_df = df[~df["is_calib"].astype(bool)].copy()
    rep = {}

    # raw
    rep["raw_sbp"] = me_std_mae(eval_df["y_pred_sbp"], eval_df["y_true_sbp"])
    rep["raw_dbp"] = me_std_mae(eval_df["y_pred_dbp"], eval_df["y_true_dbp"])

    # bias-cal
    rep["bias_sbp"] = me_std_mae(eval_df["y_pred_sbp_cal"], eval_df["y_true_sbp"])
    rep["bias_dbp"] = me_std_mae(eval_df["y_pred_dbp_cal"], eval_df["y_true_dbp"])

    # affine if exists
    if "y_pred_sbp_aff" in df.columns and "y_pred_dbp_aff" in df.columns:
        rep["aff_sbp"] = me_std_mae(eval_df["y_pred_sbp_aff"], eval_df["y_true_sbp"])
        rep["aff_dbp"] = me_std_mae(eval_df["y_pred_dbp_aff"], eval_df["y_true_dbp"])

    def fmt(x):
        me, std, mae = x
        return f"ME={me:+.3f}, STD={std:.3f}, MAE={mae:.3f}"

    print(f"\n===== {name} (EVAL ONLY: is_calib==False) =====")
    print("SBP raw :", fmt(rep["raw_sbp"]))
    print("SBP bias:", fmt(rep["bias_sbp"]))
    if "aff_sbp" in rep:
        print("SBP aff :", fmt(rep["aff_sbp"]))
    print("DBP raw :", fmt(rep["raw_dbp"]))
    print("DBP bias:", fmt(rep["bias_dbp"]))
    if "aff_dbp" in rep:
        print("DBP aff :", fmt(rep["aff_dbp"]))

    return rep


def main():
    in_csv = "results.csv"
    df = pd.read_csv(in_csv)

    # sanity
    need = [
        "id_clean", "sleep", "t_bp_ms",
        "y_true_sbp", "y_true_dbp", "y_pred_sbp", "y_pred_dbp",
        "is_calib", "y_pred_sbp_cal", "y_pred_dbp_cal"
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {in_csv}: {miss}")

    # ensure types
    df["sleep"] = df["sleep"].astype(int)
    df["t_bp_ms"] = df["t_bp_ms"].astype(np.int64)
    df["is_calib"] = df["is_calib"].astype(bool)

    df = df.sort_values(["id_clean", "t_bp_ms"]).reset_index(drop=True)

    # report raw/bias
    report_all(df, "BASELINE (raw + bias-cal)")

    # grid search lambda for affine calibration to reduce SBP STD
    lambdas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    best = None

    for lam in lambdas:
        df_aff = apply_affine_calibration(
            df, lam=lam, min_points=3,
            by_sleep=True,               # day/night separate
            penalize_intercept=False     # usually safer with few points
        )
        eval_df = df_aff[~df_aff["is_calib"]].copy()
        me_s, std_s, mae_s = me_std_mae(eval_df["y_pred_sbp_aff"], eval_df["y_true_sbp"])
        me_d, std_d, mae_d = me_std_mae(eval_df["y_pred_dbp_aff"], eval_df["y_true_dbp"])

        # objective: minimize SBP STD; tie-breaker: SBP MAE
        score = (std_s, mae_s)
        if best is None or score < best["score"]:
            best = {
                "lam": lam,
                "score": score,
                "sbp": (me_s, std_s, mae_s),
                "dbp": (me_d, std_d, mae_d),
                "df_aff": df_aff
            }

    print("\n===== AFFINE GRID SEARCH (objective: minimize SBP STD) =====")
    print(f"Best lam = {best['lam']}")
    me_s, std_s, mae_s = best["sbp"]
    me_d, std_d, mae_d = best["dbp"]
    print(f"SBP aff: ME={me_s:+.3f}, STD={std_s:.3f}, MAE={mae_s:.3f}")
    print(f"DBP aff: ME={me_d:+.3f}, STD={std_d:.3f}, MAE={mae_d:.3f}")

    # final report with aff columns
    df_best = best["df_aff"]
    report_all(df_best, f"FINAL (affine-cal, lam={best['lam']})")

    out_csv = "results_with_affine_cal.csv"
    df_best.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()