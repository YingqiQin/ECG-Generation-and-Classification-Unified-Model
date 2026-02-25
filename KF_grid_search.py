import numpy as np
import pandas as pd
from itertools import product

# -----------------------------
# KF calibrator (time-aware), identity init
# theta = [a_s, b_s, a_d, b_d]
# -----------------------------
class KalmanAffineCalibrator2DTime:
    def __init__(
        self,
        mu0=None,
        P0=None,
        Q_per_hour=None,
        R=None,
        huber_delta=20.0,
        max_abs_innov=250.0,
    ):
        if mu0 is None:
            mu0 = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)
        self.mu = np.asarray(mu0, float).copy()

        if P0 is None:
            P0 = np.diag([0.5**2, 30.0**2, 0.5**2, 30.0**2])
        self.P = np.asarray(P0, float).copy()

        if Q_per_hour is None:
            Q_per_hour = np.diag([0.0, 1e-3, 0.0, 1e-3])
        self.Q_per_hour = np.asarray(Q_per_hour, float).copy()

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

    def _predict_theta(self, ts_sec):
        if self.last_ts is None:
            self.last_ts = ts_sec
            return 0.0
        dt_hours = max((ts_sec - self.last_ts) / 3600.0, 0.0)
        self.P = self.P + self.Q_per_hour * dt_hours
        self.last_ts = ts_sec
        return dt_hours

    def predict(self, yhat, ts_sec):
        self._predict_theta(ts_sec)
        X = self.X_from_yhat(yhat)
        y_pred = X @ self.mu
        S = X @ self.P @ X.T + self.R
        std = np.sqrt(np.maximum(np.diag(S), 1e-12))
        return y_pred, std

    def update_with_cuff(self, yhat, y_true, ts_sec):
        self._predict_theta(ts_sec)
        X = self.X_from_yhat(yhat)
        y_true = np.asarray(y_true, float).reshape(2)

        y_pred = X @ self.mu
        innov = y_true - y_pred

        # extreme guard
        if np.any(np.abs(innov) > self.max_abs_innov):
            return innov, True

        # Huber-like robustification: large innov => downweight by inflating R
        R_eff = self.R.copy()
        if self.huber_delta is not None:
            norm = float(np.linalg.norm(innov))
            if norm > self.huber_delta:
                scale = self.huber_delta / (norm + 1e-12)
                R_eff = R_eff / (scale**2 + 1e-12)

        S = X @ self.P @ X.T + R_eff
        K = self.P @ X.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ innov
        self.P = (np.eye(4) - K @ X) @ self.P
        return innov, False


# -----------------------------
# calibration point selector (deterministic)
# Default: try 4 awake + 3 sleep with >=180min gap.
# If not enough, fill from remaining points while keeping gap.
# If still not enough, relax min_gap progressively.
# -----------------------------
def select_calib_indices(df_id, ts_sec, n_total=7, n_awake=4, n_sleep=3, min_gap_min=180):
    sleep = df_id["sleep"].to_numpy().astype(int)
    idx_all = np.arange(len(df_id))

    # progressive relaxation on gap to guarantee 7 points (for fair tuning)
    gap_list = [min_gap_min, 150, 120, 90, 60, 30, 0]
    for gap_min in gap_list:
        min_gap = gap_min * 60

        def ok(i, picked):
            return all(abs(ts_sec[i] - ts_sec[j]) >= min_gap for j in picked)

        idx_awake = idx_all[sleep == 0]
        idx_sleep = idx_all[sleep == 1]

        # choose evenly spaced targets within each pool
        def pick_evenly(idx_pool, n_pick):
            if n_pick <= 0 or len(idx_pool) == 0:
                return []
            if len(idx_pool) <= n_pick:
                return list(idx_pool)
            pool_ts = ts_sec[idx_pool]
            order = np.argsort(pool_ts)
            idx_sorted = idx_pool[order]
            targets = np.linspace(0, len(idx_sorted)-1, n_pick).round().astype(int)
            return [int(idx_sorted[t]) for t in targets]

        cand = pick_evenly(idx_awake, n_awake) + pick_evenly(idx_sleep, n_sleep)
        cand = sorted(set(cand), key=lambda i: ts_sec[i])

        picked = []
        for i in cand:
            if ok(i, picked):
                picked.append(i)

        # fill remaining (prefer awake then sleep then all)
        def fill(pool):
            nonlocal picked
            for i in sorted(pool, key=lambda x: ts_sec[x]):
                if i in picked:
                    continue
                if ok(i, picked):
                    picked.append(i)
                    if len(picked) >= n_total:
                        break

        fill(idx_awake)
        fill(idx_sleep)
        fill(idx_all)

        if len(picked) >= n_total:
            picked = sorted(picked[:n_total], key=lambda i: ts_sec[i])
            return np.array(picked, dtype=int)

    # fallback
    return np.array(sorted(idx_all[:min(n_total, len(idx_all))]), dtype=int)


def me_sd_mae(err):
    err = np.asarray(err, float)
    return float(err.mean()), float(err.std(ddof=0)), float(np.mean(np.abs(err)))


# -----------------------------
# Evaluate one parameter set on val CSV
# Returns micro + macro + diagnostics
# -----------------------------
def eval_params_on_val(
    val_csv_path: str,
    qa: float,
    qb: float,
    sigma_R: float,
    sigma_a0: float,
    sigma_b0: float,
    huber_delta: float,
    n_awake=4,
    n_sleep=3,
    min_gap_min=180,
    lambda_me=1.0,
):
    df = pd.read_csv(val_csv_path)

    required = ["id_clean","t_bp_ms","sleep","y_true_sbp","y_true_dbp","y_pred_sbp","y_pred_dbp"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    pooled = {"raw_s":[], "raw_d":[], "kf_s":[], "kf_d":[]}
    per_id_rows = []
    skip_updates = 0
    total_updates = 0

    for pid, g in df.groupby("id_clean", sort=False):
        g = g.copy()

        ts_sec = (g["t_bp_ms"].to_numpy(np.int64) // 1000)
        order = np.argsort(ts_sec)
        g = g.iloc[order].reset_index(drop=True)
        ts_sec = ts_sec[order]

        calib_idx = select_calib_indices(
            g, ts_sec,
            n_total=7, n_awake=n_awake, n_sleep=n_sleep,
            min_gap_min=min_gap_min
        )
        calib_mask = np.zeros(len(g), dtype=bool)
        calib_mask[calib_idx] = True

        # KF hyperparams
        Q = np.diag([qa, qb, qa, qb]).astype(float)
        R = np.diag([sigma_R**2, sigma_R**2]).astype(float)
        P0 = np.diag([sigma_a0**2, sigma_b0**2, sigma_a0**2, sigma_b0**2]).astype(float)

        kf = KalmanAffineCalibrator2DTime(P0=P0, Q_per_hour=Q, R=R, huber_delta=huber_delta)

        yhat = g[["y_pred_sbp","y_pred_dbp"]].to_numpy(float)
        ytrue= g[["y_true_sbp","y_true_dbp"]].to_numpy(float)

        ykf = np.zeros_like(yhat)

        for i in range(len(g)):
            pred, _ = kf.predict(yhat[i], int(ts_sec[i]))
            ykf[i] = pred

            if calib_mask[i]:
                innov, skipped = kf.update_with_cuff(yhat[i], ytrue[i], int(ts_sec[i]))
                total_updates += 1
                if skipped:
                    skip_updates += 1

        eval_mask = ~calib_mask
        err_raw_s = (yhat[eval_mask,0] - ytrue[eval_mask,0])
        err_raw_d = (yhat[eval_mask,1] - ytrue[eval_mask,1])
        err_kf_s  = (ykf[eval_mask,0] - ytrue[eval_mask,0])
        err_kf_d  = (ykf[eval_mask,1] - ytrue[eval_mask,1])

        pooled["raw_s"].append(err_raw_s); pooled["raw_d"].append(err_raw_d)
        pooled["kf_s"].append(err_kf_s);   pooled["kf_d"].append(err_kf_d)

        # per-id metrics (macro)
        r_s = me_sd_mae(err_raw_s); r_d = me_sd_mae(err_raw_d)
        k_s = me_sd_mae(err_kf_s);  k_d = me_sd_mae(err_kf_d)

        per_id_rows.append({
            "id_clean": pid,
            "n_points": int(len(g)),
            "n_calib": int(calib_mask.sum()),
            "RAW_SBP_ME": r_s[0], "RAW_SBP_SD": r_s[1], "RAW_SBP_MAE": r_s[2],
            "RAW_DBP_ME": r_d[0], "RAW_DBP_SD": r_d[1], "RAW_DBP_MAE": r_d[2],
            "KF_SBP_ME":  k_s[0], "KF_SBP_SD":  k_s[1], "KF_SBP_MAE":  k_s[2],
            "KF_DBP_ME":  k_d[0], "KF_DBP_SD":  k_d[1], "KF_DBP_MAE":  k_d[2],
        })

    # micro pooled
    err_raw_s_all = np.concatenate(pooled["raw_s"]) if pooled["raw_s"] else np.array([])
    err_raw_d_all = np.concatenate(pooled["raw_d"]) if pooled["raw_d"] else np.array([])
    err_kf_s_all  = np.concatenate(pooled["kf_s"])  if pooled["kf_s"]  else np.array([])
    err_kf_d_all  = np.concatenate(pooled["kf_d"])  if pooled["kf_d"]  else np.array([])

    raw_s = me_sd_mae(err_raw_s_all); raw_d = me_sd_mae(err_raw_d_all)
    kf_s  = me_sd_mae(err_kf_s_all);  kf_d  = me_sd_mae(err_kf_d_all)

    micro = {
        "RAW_SBP_ME": raw_s[0], "RAW_SBP_SD": raw_s[1], "RAW_SBP_MAE": raw_s[2],
        "RAW_DBP_ME": raw_d[0], "RAW_DBP_SD": raw_d[1], "RAW_DBP_MAE": raw_d[2],
        "KF_SBP_ME":  kf_s[0],  "KF_SBP_SD":  kf_s[1],  "KF_SBP_MAE":  kf_s[2],
        "KF_DBP_ME":  kf_d[0],  "KF_DBP_SD":  kf_d[1],  "KF_DBP_MAE":  kf_d[2],
        "N_EVAL_POINTS": int(err_kf_s_all.shape[0]),
    }

    met = pd.DataFrame(per_id_rows)
    macro = met.drop(columns=["id_clean"]).mean(numeric_only=True).to_dict()

    # objective aligned with ME±STD (lower is better)
    obj = (micro["KF_SBP_SD"] + micro["KF_DBP_SD"]) + lambda_me * (
        abs(micro["KF_SBP_ME"]) + abs(micro["KF_DBP_ME"])
    )

    diag = {
        "skip_update_ratio": (skip_updates / max(total_updates, 1)),
        "total_updates": int(total_updates),
    }

    return micro, macro, obj, diag


# -----------------------------
# Grid search driver
# -----------------------------
def grid_search_kf(val_csv_path: str, out_csv_path: str):
    # Recommended search space (start moderate)
    qa_list = [0.0, 1e-7, 1e-6, 1e-5]               # scale drift (often ~0)
    qb_list = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]   # offset drift
    sigma_R_list = [3.0, 4.0, 5.0, 6.0, 8.0]         # cuff noise std (mmHg)
    sigma_a0_list = [0.2, 0.5, 1.0]                  # initial slope uncertainty
    sigma_b0_list = [10.0, 20.0, 30.0, 50.0]         # initial offset uncertainty
    huber_delta_list = [15.0, 20.0, 30.0, 40.0]

    rows = []
    best = None

    for qa, qb, sR, sa0, sb0, hd in product(qa_list, qb_list, sigma_R_list, sigma_a0_list, sigma_b0_list, huber_delta_list):
        micro, macro, obj, diag = eval_params_on_val(
            val_csv_path=val_csv_path,
            qa=qa, qb=qb,
            sigma_R=sR,
            sigma_a0=sa0, sigma_b0=sb0,
            huber_delta=hd,
            # keep these fixed during tuning (consistent protocol)
            n_awake=4, n_sleep=3, min_gap_min=180,
            lambda_me=1.0,
        )

        row = {
            "qa": qa, "qb": qb, "sigma_R": sR, "sigma_a0": sa0, "sigma_b0": sb0, "huber_delta": hd,
            "obj": obj,
            "skip_update_ratio": diag["skip_update_ratio"],
            "N_EVAL_POINTS": micro["N_EVAL_POINTS"],
            # micro KF
            "KF_SBP_ME_micro": micro["KF_SBP_ME"],
            "KF_SBP_SD_micro": micro["KF_SBP_SD"],
            "KF_DBP_ME_micro": micro["KF_DBP_ME"],
            "KF_DBP_SD_micro": micro["KF_DBP_SD"],
            "KF_SBP_MAE_micro": micro["KF_SBP_MAE"],
            "KF_DBP_MAE_micro": micro["KF_DBP_MAE"],
            # micro RAW (for reference)
            "RAW_SBP_SD_micro": micro["RAW_SBP_SD"],
            "RAW_DBP_SD_micro": micro["RAW_DBP_SD"],
        }
        rows.append(row)

        if best is None or obj < best["obj"]:
            best = row

    res = pd.DataFrame(rows).sort_values("obj", ascending=True).reset_index(drop=True)
    res.to_csv(out_csv_path, index=False)
    return res, best


# -----------------------------
# Example usage:
# res, best = grid_search_kf("val_set.csv", "kf_grid_val_results.csv")
# print("BEST:", best)
# print(res.head(10))
# -----------------------------