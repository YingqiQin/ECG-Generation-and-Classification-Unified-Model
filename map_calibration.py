import pandas as pd


def apply_external_calib_points(
    df: pd.DataFrame,
    ext_calib_df: pd.DataFrame,
    *,
    df_id_col: str = "id_upper",
    df_time_col: str = "t_bp_ms",
    ext_id_col: str = "id",
    ext_time_col: str = "time_timestamp",
    is_calib_col: str = "is_calib",
    reset_existing: bool = True,
    drop_ext_duplicates: bool = True,
    verbose: bool = True,
):
    """
    根据外部团队给的校准点(csv里的 id + time_timestamp)，
    将本地 df 中匹配到的行标记为 is_calib=True。

    Parameters
    ----------
    df : 本地事件级 DataFrame，至少包含 [df_id_col, df_time_col]
    ext_calib_df : 外部校准点 DataFrame，至少包含 [ext_id_col, ext_time_col]
    reset_existing : True 时先把 df[is_calib_col] 全部置 False，再按外部点重打标
    drop_ext_duplicates : True 时对外部校准点按 (id, time) 去重
    """

    out = df.copy()

    # 1) 基础检查
    need_df_cols = {df_id_col, df_time_col}
    need_ext_cols = {ext_id_col, ext_time_col}
    miss_df = need_df_cols - set(out.columns)
    miss_ext = need_ext_cols - set(ext_calib_df.columns)

    if miss_df:
        raise ValueError(f"df 缺少必要列: {miss_df}")
    if miss_ext:
        raise ValueError(f"ext_calib_df 缺少必要列: {miss_ext}")

    # 2) 规范化 key
    out["_match_id"] = out[df_id_col].astype(str).str.strip().str.upper()
    out["_match_t"] = pd.to_numeric(out[df_time_col], errors="coerce").astype("Int64")

    ext = ext_calib_df[[ext_id_col, ext_time_col]].copy()
    ext["_match_id"] = ext[ext_id_col].astype(str).str.strip().str.upper()
    ext["_match_t"] = pd.to_numeric(ext[ext_time_col], errors="coerce").astype("Int64")

    # 去掉非法 key
    ext = ext.dropna(subset=["_match_id", "_match_t"]).copy()

    # 3) 外部点去重
    n_ext_raw = len(ext)
    if drop_ext_duplicates:
        ext = ext.drop_duplicates(subset=["_match_id", "_match_t"]).copy()
    n_ext_unique = len(ext)

    # 4) 构造匹配 key
    ext_keys = set(zip(ext["_match_id"].tolist(), ext["_match_t"].tolist()))
    df_keys = list(zip(out["_match_id"].tolist(), out["_match_t"].tolist()))

    # 5) 重置 / 初始化 is_calib
    if is_calib_col not in out.columns or reset_existing:
        out[is_calib_col] = False
    else:
        out[is_calib_col] = out[is_calib_col].fillna(False).astype(bool)

    # 6) 按外部 key 覆盖打标
    matched_mask = [(k in ext_keys) for k in df_keys]
    out.loc[matched_mask, is_calib_col] = True

    # 7) 统计信息
    matched_df_rows = int(sum(matched_mask))
    matched_df_unique_keys = len(set([k for k, m in zip(df_keys, matched_mask) if m]))
    unmatched_ext_keys = ext_keys - set(df_keys)

    report = {
        "n_df_rows": len(out),
        "n_ext_raw": n_ext_raw,
        "n_ext_unique": n_ext_unique,
        "n_marked_true_rows": matched_df_rows,
        "n_matched_unique_keys": matched_df_unique_keys,
        "n_unmatched_ext_keys": len(unmatched_ext_keys),
    }

    if verbose:
        print("=== apply_external_calib_points report ===")
        for k, v in report.items():
            print(f"{k}: {v}")

        if len(unmatched_ext_keys) > 0:
            print("\n前几个未匹配到本地 df 的外部校准点:")
            preview = list(unmatched_ext_keys)[:10]
            for x in preview:
                print(x)

    # 8) 清理临时列
    out = out.drop(columns=["_match_id", "_match_t"])

    return out, report

df = pd.read_csv("your_event_level.csv")
ext = pd.read_csv("their_calib_points.csv")

df_new, report = apply_external_calib_points(
    df,
    ext,
    df_id_col="id_upper",
    df_time_col="t_bp_ms",
    ext_id_col="id",
    ext_time_col="time_timestamp",
    is_calib_col="is_calib",
    reset_existing=True,   # 用外部校准点完全覆盖你原来的选点
    verbose=True,
)

df_new.to_csv("your_event_level_with_external_calib.csv", index=False)