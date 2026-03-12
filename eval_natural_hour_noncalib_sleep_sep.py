import argparse
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


def fmt(name, triple):
    me, std, mae = triple
    return f"{name}: ME={me:+.3f}, STD={std:.3f}, MAE={mae:.3f}"


def build_state_aware_natural_hour_groups(
    df_id: pd.DataFrame,
    window_minutes: int = 60,
    true_sbp_col: str = "y_true_sbp",
    true_dbp_col: str = "y_true_dbp",
    pred_sbp_col: str = "y_pred_sbp_cal",
    pred_dbp_col: str = "y_pred_dbp_cal",
):
    """
    对单个 id_clean 做“非校准 + sleep分离 + 自然小时 + 不重叠”分组：

    规则：
      - 当前未分组的第一个点作为起点 t0
      - 当前组只纳入：
          1) t <= t0 + 60min
          2) sleep 与起点相同
      - 一旦 sleep 改变，立即截断当前组
      - 下一组从第一个未使用点重新开始
    """
    window_ms = int(window_minutes * 60 * 1000)

    g = df_id.sort_values("t_bp_ms").reset_index(drop=False)  # 保留原始 index
    rows = []

    i = 0
    n = len(g)
    while i < n:
        t0 = int(g.loc[i, "t_bp_ms"])
        sleep0 = int(g.loc[i, "sleep"])
        tend = t0 + window_ms

        j = i
        while j + 1 < n:
            t_next = int(g.loc[j + 1, "t_bp_ms"])
            sleep_next = int(g.loc[j + 1, "sleep"])

            # 超过自然小时，停止
            if t_next > tend:
                break

            # sleep状态改变，停止（白天/睡眠完全分离）
            if sleep_next != sleep0:
                break

            j += 1

        block = g.loc[i:j].copy()

        rows.append({
            "id_clean": str(block["id_clean"].iloc[0]),
            "sleep": int(sleep0),
            "hour_start_ms": int(t0),
            "hour_end_ms": int(tend),
            "first_t_bp_ms": int(block["t_bp_ms"].iloc[0]),
            "last_t_bp_ms": int(block["t_bp_ms"].iloc[-1]),
            "n_event": int(len(block)),

            "y_true_sbp_h": float(block[true_sbp_col].mean()),
            "y_true_dbp_h": float(block[true_dbp_col].mean()),
            "y_pred_sbp_h": float(block[pred_sbp_col].mean()),
            "y_pred_dbp_h": float(block[pred_dbp_col].mean()),

            "event_t_bp_ms_list": ",".join(block["t_bp_ms"].astype(np.int64).astype(str).tolist()),
            "event_idx_list": ",".join(block["index"].astype(int).astype(str).tolist()),
        })

        i = j + 1

    return rows


def report_metrics(df_hourly, tag="ALL"):
    if len(df_hourly) == 0:
        print(f"\n===== {tag} =====")
        print("No rows.")
        return None

    sbp = me_std_mae(df_hourly["y_pred_sbp_h"], df_hourly["y_true_sbp_h"])
    dbp = me_std_mae(df_hourly["y_pred_dbp_h"], df_hourly["y_true_dbp_h"])

    print(f"\n===== {tag} =====")
    print(fmt("SBP", sbp))
    print(fmt("DBP", dbp))
    print("n_hourly =", len(df_hourly))
    print("n_event distribution:")
    print(df_hourly["n_event"].value_counts().sort_index())

    return {"sbp": sbp, "dbp": dbp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True, help="affine_calibrate.csv")
    ap.add_argument("--out_csv", type=str, default="hourly_noncalib_sleepsep.csv")

    ap.add_argument("--true_sbp_col", type=str, default="y_true_sbp")
    ap.add_argument("--true_dbp_col", type=str, default="y_true_dbp")
    ap.add_argument("--pred_sbp_col", type=str, default="y_pred_sbp_cal")
    ap.add_argument("--pred_dbp_col", type=str, default="y_pred_dbp_cal")

    ap.add_argument("--window_minutes", type=int, default=60)
    ap.add_argument("--min_events", type=int, default=1)
    ap.add_argument("--timezone", type=str, default="", help="例如 Asia/Shanghai")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required = [
        "id_clean", "t_bp_ms", "sleep", "is_calib",
        args.true_sbp_col, args.true_dbp_col,
        args.pred_sbp_col, args.pred_dbp_col,
    ]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {args.in_csv}: {miss}")

    df["id_clean"] = df["id_clean"].astype(str).str.strip()
    df["t_bp_ms"] = df["t_bp_ms"].astype(np.int64)
    df["sleep"] = df["sleep"].astype(int)
    df["is_calib"] = df["is_calib"].astype(bool)

    # 只保留非校准点
    df = df[~df["is_calib"]].copy().reset_index(drop=True)

    all_rows = []
    for pid, g in df.groupby("id_clean", sort=False):
        rows = build_state_aware_natural_hour_groups(
            g,
            window_minutes=args.window_minutes,
            true_sbp_col=args.true_sbp_col,
            true_dbp_col=args.true_dbp_col,
            pred_sbp_col=args.pred_sbp_col,
            pred_dbp_col=args.pred_dbp_col,
        )
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["id_clean", "hour_start_ms"]).reset_index(drop=True)

    if args.timezone and len(out) > 0:
        dt_start = pd.to_datetime(out["hour_start_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)
        dt_end = pd.to_datetime(out["hour_end_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)
        dt_first = pd.to_datetime(out["first_t_bp_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)
        dt_last = pd.to_datetime(out["last_t_bp_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)

        out["hour_start_dt"] = dt_start.astype(str)
        out["hour_end_dt"] = dt_end.astype(str)
        out["first_t_bp_dt"] = dt_first.astype(str)
        out["last_t_bp_dt"] = dt_last.astype(str)

    # 最少事件数过滤
    out_eval = out[out["n_event"] >= int(args.min_events)].copy()

    # 总体
    report_metrics(out_eval, tag=f"ALL (non-calib, sleep-separated, min_events={args.min_events})")

    # 白天
    day_df = out_eval[out_eval["sleep"] == 0].copy()
    report_metrics(day_df, tag="DAY / sleep=0")

    # 睡眠
    night_df = out_eval[out_eval["sleep"] == 1].copy()
    report_metrics(night_df, tag="SLEEP / sleep=1")

    out.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")
    print(out.head(10))


if __name__ == "__main__":
    main()