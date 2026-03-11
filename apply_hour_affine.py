import numpy as np
import pandas as pd


def me_std_mae(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = pred - true
    me = float(err.mean())
    std = float(err.std(ddof=1)) if len(err) > 1 else float("nan")
    mae = float(np.abs(err).mean())
    return me, std, mae


def apply_affine_to_hourly_raw(
    hourly_raw_csv: str,
    affine_param_csv: str,
    out_csv: str = "hourly_pred_with_affine.csv",
    by_sleep: bool = True,
    pred_sbp_col: str = "y_pred_sbp_h",
    pred_dbp_col: str = "y_pred_dbp_h",
    true_sbp_col: str = "y_true_sbp_h",
    true_dbp_col: str = "y_true_dbp_h",
):
    """
    输入：
      hourly_raw_csv: 你多窗口推理后的小时级 raw 预测结果
      affine_param_csv: 上一步拟合出来的 subject_final_affine_params.csv

    输出：
      新增 y_pred_sbp_h_aff / y_pred_dbp_h_aff
    """
    df = pd.read_csv(hourly_raw_csv)
    params = pd.read_csv(affine_param_csv)

    if by_sleep:
        if "sleep_mean" not in df.columns:
            raise ValueError("hourly_raw_csv 需要包含 sleep_mean 列，以便按 sleep bucket 选择参数")
        df["sleep_bucket"] = (df["sleep_mean"] >= 0.5).astype(int)

        params = params.rename(columns={"sleep": "sleep_bucket"})
        df = df.merge(
            params[[
                "id_clean", "sleep_bucket",
                "a_sbp_raw", "b_sbp_raw", "a_dbp_raw", "b_dbp_raw"
            ]],
            on=["id_clean", "sleep_bucket"],
            how="left"
        )
    else:
        # 只保留 subject-level 参数
        params_subj = (
            params.groupby("id_clean", as_index=False)[
                ["a_sbp_raw", "b_sbp_raw", "a_dbp_raw", "b_dbp_raw"]
            ].mean()
        )
        df = df.merge(params_subj, on="id_clean", how="left")

    # fallback
    df["a_sbp_raw"] = df["a_sbp_raw"].fillna(1.0)
    df["b_sbp_raw"] = df["b_sbp_raw"].fillna(0.0)
    df["a_dbp_raw"] = df["a_dbp_raw"].fillna(1.0)
    df["b_dbp_raw"] = df["b_dbp_raw"].fillna(0.0)

    df["y_pred_sbp_h_aff"] = df["a_sbp_raw"] * df[pred_sbp_col] + df["b_sbp_raw"]
    df["y_pred_dbp_h_aff"] = df["a_dbp_raw"] * df[pred_dbp_col] + df["b_dbp_raw"]

    sbp = me_std_mae(df["y_pred_sbp_h_aff"], df[true_sbp_col])
    dbp = me_std_mae(df["y_pred_dbp_h_aff"], df[true_dbp_col])

    print("===== HOURLY AFTER FINAL AFFINE =====")
    print(f"SBP: ME={sbp[0]:+.3f}, STD={sbp[1]:.3f}, MAE={sbp[2]:.3f}")
    print(f"DBP: ME={dbp[0]:+.3f}, STD={dbp[1]:.3f}, MAE={dbp[2]:.3f}")

    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    return df, sbp, dbp


if __name__ == "__main__":
    apply_affine_to_hourly_raw(
        hourly_raw_csv="hourly_pred_raw.csv",
        affine_param_csv="subject_final_affine_params.csv",
        out_csv="hourly_pred_with_affine.csv",
        by_sleep=True,
    )