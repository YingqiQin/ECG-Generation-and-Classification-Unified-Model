import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def build_natural_hour_groups(df_id: pd.DataFrame, window_minutes: int = 60):
    """
    对单个 id_clean 的事件表做“自然小时、非重叠”分组。
    规则：
      - 取当前未分组的第一个点作为起点 t0
      - 收集所有 t <= t0 + 60min 的连续后续点
      - 下一组从第一个未被使用的点重新开始
    """
    window_ms = window_minutes * 60 * 1000

    g = df_id.sort_values("t_bp_ms").reset_index(drop=False)  # 保留原始 index
    rows = []

    i = 0
    n = len(g)
    while i < n:
        t0 = int(g.loc[i, "t_bp_ms"])
        tend = t0 + window_ms

        j = i
        while j + 1 < n and int(g.loc[j + 1, "t_bp_ms"]) <= tend:
            j += 1

        block = g.loc[i:j].copy()

        # 路径一致性检查（理论上同一个 id 应该一致）
        ppg_paths = block["ppg_path"].dropna().astype(str).unique().tolist()
        time_paths = block["time_path"].dropna().astype(str).unique().tolist()

        if len(ppg_paths) == 0:
            ppg_path = ""
        elif len(ppg_paths) == 1:
            ppg_path = ppg_paths[0]
        else:
            # 如果同一组里出现多个 path，默认取第一条，同时保留 warning 信息
            ppg_path = ppg_paths[0]

        if len(time_paths) == 0:
            time_path = ""
        elif len(time_paths) == 1:
            time_path = time_paths[0]
        else:
            time_path = time_paths[0]

        rows.append({
            "id_clean": str(block["id_clean"].iloc[0]),
            "hour_start_ms": int(t0),
            "hour_end_ms": int(tend),
            "first_t_bp_ms": int(block["t_bp_ms"].iloc[0]),
            "last_t_bp_ms": int(block["t_bp_ms"].iloc[-1]),
            "n_event": int(len(block)),
            "y_true_sbp_h": float(block["sbp"].mean()),
            "y_true_dbp_h": float(block["dbp"].mean()),
            "sleep_mean": float(block["sleep"].mean()),
            "ppg_path": ppg_path,
            "time_path": time_path,
            "event_t_bp_ms_list": ",".join(block["t_bp_ms"].astype(np.int64).astype(str).tolist()),
            "event_idx_list": ",".join(block["index"].astype(int).astype(str).tolist()),
        })

        i = j + 1

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True, help="你的 test_csv")
    ap.add_argument("--out_csv", type=str, required=True, help="输出 hourly_gt.csv")
    ap.add_argument("--window_minutes", type=int, default=60)
    ap.add_argument("--timezone", type=str, default="", help="可选：仅用于输出可读时间列，例如 Asia/Shanghai")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    required_cols = [
        "id_clean", "t_bp_ms", "sleep", "sbp", "dbp",
        "time_path", "ppg_path"
    ]
    miss = [c for c in required_cols if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")

    # 基本清洗
    df = df.copy()
    df["id_clean"] = df["id_clean"].astype(str).str.strip()
    df["t_bp_ms"] = df["t_bp_ms"].astype(np.int64)
    df["sbp"] = df["sbp"].astype(float)
    df["dbp"] = df["dbp"].astype(float)
    df["sleep"] = df["sleep"].astype(float)

    all_rows = []
    for pid, g in df.groupby("id_clean", sort=False):
        rows = build_natural_hour_groups(g, window_minutes=args.window_minutes)
        all_rows.extend(rows)

    out = pd.DataFrame(all_rows)

    # 可选：加人类可读时间列，方便你检查
    if args.timezone:
        dt_start = pd.to_datetime(out["hour_start_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)
        dt_end = pd.to_datetime(out["hour_end_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)
        dt_first = pd.to_datetime(out["first_t_bp_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)
        dt_last = pd.to_datetime(out["last_t_bp_ms"], unit="ms", utc=True).dt.tz_convert(args.timezone)

        out["hour_start_dt"] = dt_start.astype(str)
        out["hour_end_dt"] = dt_end.astype(str)
        out["first_t_bp_dt"] = dt_first.astype(str)
        out["last_t_bp_dt"] = dt_last.astype(str)

    out = out.sort_values(["id_clean", "hour_start_ms"]).reset_index(drop=True)
    out.to_csv(args.out_csv, index=False)

    print("===== DONE =====")
    print(f"input rows  : {len(df)}")
    print(f"hourly rows : {len(out)}")
    print(f"saved to    : {args.out_csv}")
    print(out.head(10))


if __name__ == "__main__":
    main()