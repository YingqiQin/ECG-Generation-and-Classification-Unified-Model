import numpy as np
import pandas as pd


def fit_affine_ridge(x, y, lam=10.0, penalize_intercept=False):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.stack([x, np.ones_like(x)], axis=1)  # [n,2] => [a,b]
    R = np.diag([lam, lam if penalize_intercept else 0.0])
    w = np.linalg.solve(X.T @ X + R, X.T @ y)
    a, b = float(w[0]), float(w[1])
    return a, b


def build_final_raw_affine_params(
    calib_csv: str,
    out_csv: str = "subject_final_affine_params.csv",
    lam: float = 10.0,
    min_points: int = 3,
    by_sleep: bool = True,
    penalize_intercept: bool = False,
    pred_sbp_col: str = "y_pred_sbp_cal",   # 这里用 bias 后的预测
    pred_dbp_col: str = "y_pred_dbp_cal",
    true_sbp_col: str = "y_true_sbp",
    true_dbp_col: str = "y_true_dbp",
    clip_a=(0.2, 5.0),   # 防止极端 slope
):
    """
    从 calib.csv 里拟合：
      y_true ≈ a * y_pred_biascal + b
    再结合 bias offset，合成为最终作用在 raw 预测上的：
      y_final = a_raw * y_raw + b_raw
    其中：
      a_raw = a
      b_raw = a * off + b
    """
    df = pd.read_csv(calib_csv)

    need = [
        "id_clean", "sleep", "is_calib",
        true_sbp_col, true_dbp_col,
        pred_sbp_col, pred_dbp_col,
        "off_sbp_use", "off_dbp_use"
    ]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {calib_csv}: {miss}")

    df["sleep"] = df["sleep"].astype(int)
    df["is_calib"] = df["is_calib"].astype(bool)
    calib = df[df["is_calib"]].copy()

    rows = []

    if by_sleep:
        groups = df.groupby(["id_clean", "sleep"], sort=False)
    else:
        groups = df.groupby(["id_clean"], sort=False)

    subj_pool = {pid: g for pid, g in calib.groupby("id_clean", sort=False)}

    for key, g_all in groups:
        if by_sleep:
            pid, slp = key
            g_cal = calib[(calib["id_clean"] == pid) & (calib["sleep"] == slp)]
        else:
            pid = key
            slp = -1
            g_cal = calib[calib["id_clean"] == pid]

        g_cal_subj = subj_pool.get(pid, None)

        # 这个 group 对应的 bias offset（应当基本恒定）
        off_sbp = float(g_all["off_sbp_use"].dropna().iloc[0]) if g_all["off_sbp_use"].notna().any() else 0.0
        off_dbp = float(g_all["off_dbp_use"].dropna().iloc[0]) if g_all["off_dbp_use"].notna().any() else 0.0

        def fit_or_fallback(pred_col, true_col):
            if g_cal is not None and len(g_cal) >= min_points:
                a, b = fit_affine_ridge(
                    g_cal[pred_col], g_cal[true_col],
                    lam=lam, penalize_intercept=penalize_intercept
                )
                src = "group"
                npt = len(g_cal)
            elif g_cal_subj is not None and len(g_cal_subj) >= min_points:
                a, b = fit_affine_ridge(
                    g_cal_subj[pred_col], g_cal_subj[true_col],
                    lam=lam, penalize_intercept=penalize_intercept
                )
                src = "subject_fallback"
                npt = len(g_cal_subj)
            else:
                a, b = 1.0, 0.0
                src = "identity"
                npt = 0

            # clip slope
            if clip_a is not None:
                a = float(np.clip(a, clip_a[0], clip_a[1]))
            return a, b, src, npt

        a_s, b_s, src_s, npt_s = fit_or_fallback(pred_sbp_col, true_sbp_col)
        a_d, b_d, src_d, npt_d = fit_or_fallback(pred_dbp_col, true_dbp_col)

        # 合成为直接作用在 raw prediction 上的 affine
        # y_final = a * (y_raw + off) + b = a*y_raw + (a*off + b)
        b_raw_s = a_s * off_sbp + b_s
        b_raw_d = a_d * off_dbp + b_d

        rows.append({
            "id_clean": pid,
            "sleep": int(slp),
            "off_sbp": off_sbp,
            "off_dbp": off_dbp,

            "a_sbp_biasspace": a_s,
            "b_sbp_biasspace": b_s,
            "a_dbp_biasspace": a_d,
            "b_dbp_biasspace": b_d,

            "a_sbp_raw": a_s,
            "b_sbp_raw": b_raw_s,
            "a_dbp_raw": a_d,
            "b_dbp_raw": b_raw_d,

            "src_sbp": src_s,
            "src_dbp": src_d,
            "npt_sbp": npt_s,
            "npt_dbp": npt_d,
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(out.head())
    return out


if __name__ == "__main__":
    # 示例
    build_final_raw_affine_params(
        calib_csv="results_with_bias_cal.csv",
        out_csv="subject_final_affine_params.csv",
        lam=100.0,
        min_points=3,
        by_sleep=False,
        penalize_intercept=False,
    )