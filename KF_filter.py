import numpy as np
import pandas as pd

# -----------------------------
# 1) Time-aware Kalman affine calibrator (SBP/DBP jointly, 4 params)
# theta = [a_s, b_s, a_d, b_d]
# y = X(yhat) theta + eps
# theta_t = theta_{t-1} + eta,  Cov(eta)=Q_per_hour * dt_hours
# -----------------------------
class KalmanAffineCalibrator2DTime:
    def __init__(
        self,
        mu0=None,
        P0=None,
        Q_per_hour=None,
        R=None,
        huber_delta=20.0,
        max_abs_innov=60.0,
    ):
        self.mu = np.zeros(4, dtype=float) if mu0 is None else np.asarray(mu0, float).copy()
        self.P  = np.eye(4, dtype=float) * 1.0 if P0 is None else np.asarray(P0, float).copy()

        # Drift strength per hour (tune on val later)
        if Q_per_hour is None:
            # offsets drift more than slopes (reasonable default)
            Q_per_hour = np.diag([1e-5, 1e-2, 1e-5, 1e-2])
        self.Q_per_hour = np.asarray(Q_per_hour, float).copy()

        # Cuff measurement noise (SBP/DBP), start with 4 mmHg std
        if R is None:
            R = np.diag([4.0**2, 4.0**2])
        self.R = np.asarray(R, float).copy()

        self.huber_delta = float(huber_delta)
        self.max_abs_innov = float(max_abs_innov)

        self.last_ts = None

    @staticmethod
    def X_from_yhat(yhat):
        sh, dh = float(yhat[0]), float(yhat[1])
        return np.array([[sh, 1.0, 0.0, 0.0],
                         [0.0, 0.0, dh, 1.0]], dtype=float)

    def _predict_theta(self, ts):
        if self.last_ts is None:
            self.last_ts = ts
            return 0.0
        dt_sec = float(ts - self.last_ts)
        dt_hours = max(dt_sec / 3600.0, 0.0)
        self.P = self.P + self.Q_per_hour * dt_hours
        self.last_ts = ts
        return dt_hours

    def predict(self, yhat, ts):
        self._predict_theta(ts)
        X = self.X_from_yhat(yhat)
        y_pred = X @ self.mu
        S = X @ self.P @ X.T + self.R
        std = np.sqrt(np.maximum(np.diag(S), 1e-12))
        return y_pred, std

    def update_with_cuff(self, yhat, y_true, ts):
        self._predict_theta(ts)

        X = self.X_from_yhat(yhat)
        y_true = np.asarray(y_true, float).reshape(2)

        y_pred = X @ self.mu
        innov = y_true - y_pred

        # skip absurd outliers (alignment / cuff glitch)
        if np.any(np.abs(innov) > self.max_abs_innov):
            return innov, True

        # Huber-like robust: if innovation huge, downweight by inflating R
        R_eff = self.R.copy()
        norm = float(np.linalg.norm(innov))
        if self.huber_delta is not None and norm > self.huber_delta:
            scale = self.huber_delta / (norm + 1e-12)  # (0,1]
            R_eff = R_eff / (scale**2 + 1e-12)

        S = X @ self.P @ X.T + R_eff
        K = self.P @ X.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ innov
        self.P = (np.eye(4) - K @ X) @ self.P
        return innov, False


# -----------------------------
# 2) Utilities: timestamp handling
# -----------------------------
def infer_ts_column(df: pd.DataFrame):
    # common candidates
    cands = ["timestamp", "ts", "time", "datetime", "date_time", "record_time"]
    for c in cands:
        if c in df.columns:
            return c
    return None

def build_ts_seconds(df: pd.DataFrame, ts_col: str | None, default_step_sec: int = 1800):
    """
    Return np array of ts in seconds (monotonic within each id).
    - If ts_col exists: parse to datetime if needed
    - Else: use row index * default_step_sec
    """
    if ts_col is None:
        # equal-spacing fallback
        return (np.arange(len(df), dtype=np.int64) * default_step_sec).astype(np.int64)

    s = df[ts_col]
    if np.issubdtype(s.dtype, np.number):
        # assume it's already seconds (or ms); normalize to seconds if too large
        v = s.to_numpy().astype(np.int64)
        if v.max() > 10_000_000_000:  # likely ms
            v = (v // 1000).astype(np.int64)
        return v

    # parse datetime strings
    dt = pd.to_datetime(s, errors="coerce")
    if dt.isna().any():
        raise ValueError(f"Cannot parse ts_col={ts_col} to datetime for some rows.")
    # convert to unix seconds
    return (dt.view("int64") // 10**9).astype(np.int64)


# -----------------------------
# 3) Select 7 calibration points per id with constraints:
#    - 4 awake (sleep==0), 3 sleep (sleep==1)
#    - min gap >= 180 min between any two selected points
#    - choose roughly evenly spaced in time (greedy on target positions)
# -----------------------------
def select_calibration_mask_one_id(df_id: pd.DataFrame, ts_sec: np.ndarray,
                                   n_awake=4, n_sleep=3, min_gap_min=180):
    min_gap = min_gap_min * 60

    idx_all = np.arange(len(df_id))
    idx_awake = idx_all[df_id["sleep"].to_numpy().astype(int) == 0]
    idx_sleep = idx_all[df_id["sleep"].to_numpy().astype(int) == 1]

    def pick_evenly(idx_pool, n_pick):
        if len(idx_pool) == 0 or n_pick <= 0:
            return []
        if len(idx_pool) <= n_pick:
            return list(idx_pool)

        # target positions across this subset (by time order)
        pool_ts = ts_sec[idx_pool]
        order = np.argsort(pool_ts)
        idx_pool_sorted = idx_pool[order]
        # choose target ranks
        targets = np.linspace(0, len(idx_pool_sorted) - 1, n_pick).round().astype(int)
        return [int(idx_pool_sorted[t]) for t in targets]

    # initial proposals (may violate min_gap; we will fix greedily)
    cand = pick_evenly(idx_awake, n_awake) + pick_evenly(idx_sleep, n_sleep)
    cand = sorted(set(cand), key=lambda i: ts_sec[i])

    picked = []
    for i in cand:
        if all(abs(ts_sec[i] - ts_sec[j]) >= min_gap for j in picked):
            picked.append(i)

    # If not enough, fill greedily from remaining points (prefer awake then sleep)
    def fill_from_pool(idx_pool, need):
        nonlocal picked
        if need <= 0:
            return 0
        pool_sorted = sorted(idx_pool, key=lambda i: ts_sec[i])
        added = 0
        for i in pool_sorted:
            if i in picked:
                continue
            if all(abs(ts_sec[i] - ts_sec[j]) >= min_gap for j in picked):
                picked.append(i)
                added += 1
                if added >= need:
                    break
        return added

    # enforce counts as best-effort under constraints
    # count current
    def count_awake_sleep(picked_list):
        sl = df_id["sleep"].to_numpy().astype(int)
        awake = sum(sl[i] == 0 for i in picked_list)
        sleep = sum(sl[i] == 1 for i in picked_list)
        return awake, sleep

    awake_cnt, sleep_cnt = count_awake_sleep(picked)
    add_awake = max(0, n_awake - awake_cnt)
    add_sleep = max(0, n_sleep - sleep_cnt)

    fill_from_pool(idx_awake, add_awake)
    fill_from_pool(idx_sleep, add_sleep)

    # If still < 7 (constraints too strict), fill from all remaining (rare)
    if len(picked) < (n_awake + n_sleep):
        fill_from_pool(idx_all, (n_awake + n_sleep) - len(picked))

    picked = sorted(picked, key=lambda i: ts_sec[i])
    mask = np.zeros(len(df_id), dtype=bool)
    mask[picked[: (n_awake + n_sleep)]] = True
    return mask


# -----------------------------
# 4) Run KF calibration for one id (time-forward)
#    Updates only on calibration points (mask==True)
# -----------------------------
def run_kf_one_id(df_id: pd.DataFrame, ts_sec: np.ndarray, calib_mask: np.ndarray,
                  Q_per_hour=None, R=None, huber_delta=20.0):
    kf = KalmanAffineCalibrator2DTime(Q_per_hour=Q_per_hour, R=R, huber_delta=huber_delta)

    ycal = np.zeros((len(df_id), 2), dtype=float)
    ystd = np.zeros((len(df_id), 2), dtype=float)
    innov = np.full((len(df_id), 2), np.nan, dtype=float)
    theta = np.zeros((len(df_id), 4), dtype=float)
    skipped = np.zeros(len(df_id), dtype=bool)

    yhat = df_id[["y_pred_sbp", "y_pred_dbp"]].to_numpy(dtype=float)
    ytrue = df_id[["y_true_sbp", "y_true_dbp"]].to_numpy(dtype=float)

    for i in range(len(df_id)):
        pred, std = kf.predict(yhat[i], int(ts_sec[i]))
        ycal[i] = pred
        ystd[i] = std
        theta[i] = kf.mu.copy()

        if calib_mask[i]:
            inn, sk = kf.update_with_cuff(yhat[i], ytrue[i], int(ts_sec[i]))
            innov[i] = inn
            skipped[i] = sk

    return ycal, ystd, innov, theta, skipped


# -----------------------------
# 5) Metrics
# -----------------------------
def me_std_mae(err: np.ndarray):
    me = float(np.mean(err))
    sd = float(np.std(err, ddof=0))
    mae = float(np.mean(np.abs(err)))
    return me, sd, mae


# -----------------------------
# 6) Main: load csv, groupby id_clean, run, save
# -----------------------------
def calibrate_csv(
    in_csv: str,
    out_csv: str,
    min_gap_min=180,
    n_awake=4,
    n_sleep=3,
    default_step_sec=1800,
    # KF hyperparams (tune on val later)
    Q_per_hour=None,
    R=None,
    huber_delta=20.0,
):
    df = pd.read_csv(in_csv)

    required = ["id_clean", "sleep", "y_true_sbp", "y_true_dbp", "y_pred_sbp", "y_pred_dbp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ts_col = infer_ts_column(df)
    # We'll process per-id; each id is already time-sorted, but we sort anyway for safety if ts exists.
    out_frames = []
    all_metrics = []

    for pid, g in df.groupby("id_clean", sort=False):
        g = g.copy().reset_index(drop=True)

        ts_sec = build_ts_seconds(g, ts_col, default_step_sec=default_step_sec)
        if ts_col is not None:
            # ensure sorted by time
            order = np.argsort(ts_sec)
            g = g.iloc[order].reset_index(drop=True)
            ts_sec = ts_sec[order]

        calib_mask = select_calibration_mask_one_id(
            g, ts_sec,
            n_awake=n_awake, n_sleep=n_sleep,
            min_gap_min=min_gap_min
        )

        ycal, ystd, innov, theta, skipped = run_kf_one_id(
            g, ts_sec, calib_mask,
            Q_per_hour=Q_per_hour, R=R, huber_delta=huber_delta
        )

        g["is_calib7"] = calib_mask.astype(int)
        g["y_kf_sbp"] = ycal[:, 0]
        g["y_kf_dbp"] = ycal[:, 1]
        g["std_kf_sbp"] = ystd[:, 0]
        g["std_kf_dbp"] = ystd[:, 1]
        g["innov_sbp"] = innov[:, 0]
        g["innov_dbp"] = innov[:, 1]
        g["kf_a_s"] = theta[:, 0]
        g["kf_b_s"] = theta[:, 1]
        g["kf_a_d"] = theta[:, 2]
        g["kf_b_d"] = theta[:, 3]
        g["kf_skip_update"] = skipped.astype(int)

        # metrics: evaluate on NON-calibration points (time-forward, no leakage)
        eval_mask = ~calib_mask
        err_raw_s = (g.loc[eval_mask, "y_pred_sbp"] - g.loc[eval_mask, "y_true_sbp"]).to_numpy(float)
        err_raw_d = (g.loc[eval_mask, "y_pred_dbp"] - g.loc[eval_mask, "y_true_dbp"]).to_numpy(float)
        err_kf_s  = (g.loc[eval_mask, "y_kf_sbp"]   - g.loc[eval_mask, "y_true_sbp"]).to_numpy(float)
        err_kf_d  = (g.loc[eval_mask, "y_kf_dbp"]   - g.loc[eval_mask, "y_true_dbp"]).to_numpy(float)

        m_raw_s = me_std_mae(err_raw_s); m_raw_d = me_std_mae(err_raw_d)
        m_kf_s  = me_std_mae(err_kf_s);  m_kf_d  = me_std_mae(err_kf_d)

        all_metrics.append({
            "id_clean": pid,
            "n_points": int(len(g)),
            "n_calib": int(calib_mask.sum()),
            "RAW_SBP_ME": m_raw_s[0], "RAW_SBP_SD": m_raw_s[1], "RAW_SBP_MAE": m_raw_s[2],
            "RAW_DBP_ME": m_raw_d[0], "RAW_DBP_SD": m_raw_d[1], "RAW_DBP_MAE": m_raw_d[2],
            "KF_SBP_ME":  m_kf_s[0],  "KF_SBP_SD":  m_kf_s[1],  "KF_SBP_MAE":  m_kf_s[2],
            "KF_DBP_ME":  m_kf_d[0],  "KF_DBP_SD":  m_kf_d[1],  "KF_DBP_MAE":  m_kf_d[2],
        })

        out_frames.append(g)

    df_out = pd.concat(out_frames, axis=0, ignore_index=True)
    df_out.to_csv(out_csv, index=False)

    met = pd.DataFrame(all_metrics)
    # macro average over ids (simple mean)
    macro = met.drop(columns=["id_clean"]).mean(numeric_only=True).to_dict()
    return df_out, met, macro


# -----------------------------
# Example usage:
# df_out, met, macro = calibrate_csv(
#     in_csv="test_pred.csv",
#     out_csv="test_pred_kf.csv",
#     min_gap_min=180,
#     n_awake=4, n_sleep=3,
#     default_step_sec=1800, # used only if no timestamp column exists
#     Q_per_hour=np.diag([1e-5, 1e-2, 1e-5, 1e-2]),
#     R=np.diag([4.0**2, 4.0**2]),
#     huber_delta=20.0,
# )
# print(macro)
# met.to_csv("kf_metrics_per_id.csv", index=False)
# -----------------------------