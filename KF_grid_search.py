import numpy as np
import pandas as pd
from itertools import product

# -----------------------------
# Utils
# -----------------------------
def me_sd_mae(err):
    err = np.asarray(err, float)
    return float(err.mean()), float(err.std(ddof=0)), float(np.mean(np.abs(err)))

def ridge_fit_affine(yhat, ytrue, lam=1e-2, fit_intercept=True):
    """
    Fit ytrue ≈ a*yhat + b with ridge.
    Returns (a,b).
    """
    yhat = np.asarray(yhat, float).reshape(-1, 1)
    ytrue = np.asarray(ytrue, float).reshape(-1, 1)
    if fit_intercept:
        X = np.concatenate([yhat, np.ones_like(yhat)], axis=1)  # [n,2]
    else:
        X = yhat

    I = np.eye(X.shape[1], dtype=float)
    theta = np.linalg.solve(X.T @ X + lam * I, X.T @ ytrue)     # [2,1]
    if fit_intercept:
        a, b = float(theta[0, 0]), float(theta[1, 0])
        return a, b
    else:
        a = float(theta[0, 0])
        return a, 0.0

def select_calib_indices(df_id, ts_sec, n_total=7, n_awake=4, n_sleep=3, min_gap_min=180):
    """
    Deterministic selector. To ensure fair tuning, we guarantee n_total points by progressively relaxing min_gap.
    """
    sleep = df_id["sleep"].to_numpy().astype(int)
    idx_all = np.arange(len(df_id))

    gap_list = [min_gap_min, 150, 120, 90, 60, 30, 0]
    for gap_min in gap_list:
        min_gap = gap_min * 60

        def ok(i, picked):
            return all(abs(ts_sec[i] - ts_sec[j]) >= min_gap for j in picked)

        idx_awake = idx_all[sleep == 0]
        idx_sleep = idx_all[sleep == 1]

        def pick_evenly(idx_pool, n_pick):
            if n_pick <= 0 or len(idx_pool) == 0:
                return []
            if len(idx_pool) <= n_pick:
                return list(idx_pool)
            order = np.argsort(ts_sec[idx_pool])
            idx_sorted = idx_pool[order]
            targets = np.linspace(0, len(idx_sorted)-1, n_pick).round().astype(int)
            return [int(idx_sorted[t]) for t in targets]

        cand = pick_evenly(idx_awake, n_awake) + pick_evenly(idx_sleep, n_sleep)
        cand = sorted(set(cand), key=lambda i: ts_sec[i])

        picked = []
        for i in cand:
            if ok(i, picked):
                picked.append(i)

        def fill(pool):
            nonlocal picked
            for i in sorted(pool, key=lambda x: ts_sec[x]):
                if i in picked:
                    continue
                if ok(i, picked):
                    picked.append(i)
                    if len(picked) >= n_total:
                        break

        fill(idx_awake); fill(idx_sleep); fill(idx_all)

        if len(picked) >= n_total:
            picked = sorted(picked[:n_total], key=lambda i: ts_sec[i])
            return np.array(picked, dtype=int)

    return np.array(sorted(idx_all[:min(n_total, len(idx_all))]), dtype=int)


# -----------------------------
# Layer-2: Kalman filter tracking ONLY offsets b_s(t), b_d(t)
# z_t = y_true - a*yhat  = b(t) + noise
# state: b = [b_s, b_d]
# -----------------------------
class OffsetKalman2DTime:
    def __init__(self, b0, P0, Qb_per_hour, R, huber_delta=20.0, max_abs_innov=250.0):
        self.b = np.asarray(b0, float).reshape(2)               # [2]
        self.P = np.asarray(P0, float).copy()                   # [2,2]
        self.Qb_per_hour = np.asarray(Qb_per_hour, float).copy()# [2,2]
        self.R = np.asarray(R, float).copy()                    # [2,2]
        self.huber_delta = float(huber_delta)
        self.max_abs_innov = float(max_abs_innov)
        self.last_ts = None

    def _predict(self, ts_sec):
        if self.last_ts is None:
            self.last_ts = ts_sec
            return 0.0
        dt_hours = max((ts_sec - self.last_ts) / 3600.0, 0.0)
        self.P = self.P + self.Qb_per_hour * dt_hours
        self.last_ts = ts_sec
        return dt_hours

    def predict_b(self, ts_sec):
        self._predict(ts_sec)
        return self.b.copy(), self.P.copy()

    def update(self, z_obs, ts_sec):
        """
        z_obs: observed offset = y_true - a*yhat, shape (2,)
        """
        self._predict(ts_sec)
        z_obs = np.asarray(z_obs, float).reshape(2)

        innov = z_obs - self.b  # H=I
        if np.any(np.abs(innov) > self.max_abs_innov):
            return innov, True

        R_eff = self.R.copy()
        if self.huber_delta is not None:
            norm = float(np.linalg.norm(innov))
            if norm > self.huber_delta:
                scale = self.huber_delta / (norm + 1e-12)
                R_eff = R_eff / (scale**2 + 1e-12)

        S = self.P + R_eff
        K = self.P @ np.linalg.inv(S)
        self.b = self.b + K @ innov
        self.P = (np.eye(2) - K) @ self.P
        return innov, False


# -----------------------------
# Run two-layer calibration on ONE id
# -----------------------------
def run_two_layer_one_id(
    g: pd.DataFrame,
    calib_idx: np.ndarray,
    lam_ridge: float,
    Qb_per_hour_diag: tuple[float, float],
    sigma_R: float,
    sigma_b0: float,
    huber_delta: float,
    clip_a: tuple[float, float] | None = None,  # e.g., (0, 3) if you want monotonic
):
    # sort by time
    ts_sec = (g["t_bp_ms"].to_numpy(np.int64) // 1000)
    order = np.argsort(ts_sec)
    g = g.iloc[order].reset_index(drop=True)
    ts_sec = ts_sec[order]

    # rebuild calib mask after sorting
    calib_mask = np.zeros(len(g), dtype=bool)
    # NOTE: calib_idx should be indices AFTER sorting, so we compute it outside using the sorted g/ts
    calib_mask[calib_idx] = True

    yhat = g[["y_pred_sbp", "y_pred_dbp"]].to_numpy(float)
    ytrue = g[["y_true_sbp", "y_true_dbp"]].to_numpy(float)

    # -------- Layer-1: day-level ridge affine on calib points --------
    idx = calib_idx
    a_s, b_s0 = ridge_fit_affine(yhat[idx, 0], ytrue[idx, 0], lam=lam_ridge)
    a_d, b_d0 = ridge_fit_affine(yhat[idx, 1], ytrue[idx, 1], lam=lam_ridge)

    if clip_a is not None:
        a_s = float(np.clip(a_s, clip_a[0], clip_a[1]))
        a_d = float(np.clip(a_d, clip_a[0], clip_a[1]))

    # initial offset state uses ridge intercepts
    b0 = np.array([b_s0, b_d0], float)
    P0 = np.diag([sigma_b0**2, sigma_b0**2]).astype(float)
    Qb = np.diag([Qb_per_hour_diag[0], Qb_per_hour_diag[1]]).astype(float)
    R = np.diag([sigma_R**2, sigma_R**2]).astype(float)

    kf = OffsetKalman2DTime(b0=b0, P0=P0, Qb_per_hour=Qb, R=R, huber_delta=huber_delta)

    # outputs
    y_cal = np.zeros_like(yhat)
    b_track = np.zeros((len(g), 2), float)
    innov = np.full((len(g), 2), np.nan, float)
    skipped = np.zeros(len(g), bool)

    for i in range(len(g)):
        # predict calibrated BP
        b_i, _ = kf.predict_b(int(ts_sec[i]))
        b_track[i] = b_i
        y_cal[i, 0] = a_s * yhat[i, 0] + b_i[0]
        y_cal[i, 1] = a_d * yhat[i, 1] + b_i[1]

        if calib_mask[i]:
            z_obs = np.array([ytrue[i, 0] - a_s * yhat[i, 0],
                              ytrue[i, 1] - a_d * yhat[i, 1]], float)
            inn, sk = kf.update(z_obs, int(ts_sec[i]))
            innov[i] = inn
            skipped[i] = sk

    return {
        "g_sorted": g,
        "order": order,
        "ts_sec": ts_sec,
        "calib_mask": calib_mask,
        "y_cal": y_cal,
        "a_s": a_s, "a_d": a_d,
        "b0_s": b_s0, "b0_d": b_d0,
        "b_track": b_track,
        "innov": innov,
        "skipped": skipped,
    }

def run_static_ridge_one_id(g_sorted, calib_idx, lam_ridge, clip_a=None):
    ts_sec = (g_sorted["t_bp_ms"].to_numpy(np.int64) // 1000)
    yhat = g_sorted[["y_pred_sbp", "y_pred_dbp"]].to_numpy(float)
    ytrue= g_sorted[["y_true_sbp", "y_true_dbp"]].to_numpy(float)

    idx = calib_idx
    a_s, b_s = ridge_fit_affine(yhat[idx,0], ytrue[idx,0], lam=lam_ridge)
    a_d, b_d = ridge_fit_affine(yhat[idx,1], ytrue[idx,1], lam=lam_ridge)

    if clip_a is not None:
        a_s = float(np.clip(a_s, clip_a[0], clip_a[1]))
        a_d = float(np.clip(a_d, clip_a[0], clip_a[1]))

    y_cal = np.zeros_like(yhat)
    y_cal[:,0] = a_s * yhat[:,0] + b_s
    y_cal[:,1] = a_d * yhat[:,1] + b_d
    return y_cal, (a_s,b_s,a_d,b_d)

def eval_params_two_layer_on_val(
    val_csv_path: str,
    lam_ridge: float,
    qb_s: float,
    qb_d: float,
    sigma_R: float,
    sigma_b0: float,
    huber_delta: float,
    n_awake=4,
    n_sleep=3,
    min_gap_min=180,
    clip_a=None,
):
    df = pd.read_csv(val_csv_path)
    required = ["id_clean","t_bp_ms","sleep","y_true_sbp","y_true_dbp","y_pred_sbp","y_pred_dbp"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    pooled = {"raw_s":[], "raw_d":[], "static_s":[], "static_d":[], "two_s":[], "two_d":[]}
    per_id_rows = []

    for pid, g in df.groupby("id_clean", sort=False):
        g = g.copy()
        ts_sec = (g["t_bp_ms"].to_numpy(np.int64) // 1000)
        order = np.argsort(ts_sec)
        g = g.iloc[order].reset_index(drop=True)
        ts_sec = ts_sec[order]

        calib_idx = select_calib_indices(
            g, ts_sec, n_total=7,
            n_awake=n_awake, n_sleep=n_sleep,
            min_gap_min=min_gap_min
        )
        calib_mask = np.zeros(len(g), bool); calib_mask[calib_idx] = True
        eval_mask = ~calib_mask

        # raw errors
        yhat = g[["y_pred_sbp","y_pred_dbp"]].to_numpy(float)
        ytrue= g[["y_true_sbp","y_true_dbp"]].to_numpy(float)
        err_raw_s = yhat[eval_mask,0] - ytrue[eval_mask,0]
        err_raw_d = yhat[eval_mask,1] - ytrue[eval_mask,1]

        # static ridge baseline
        y_static, _ = run_static_ridge_one_id(g, calib_idx, lam_ridge=lam_ridge, clip_a=clip_a)
        err_sta_s = y_static[eval_mask,0] - ytrue[eval_mask,0]
        err_sta_d = y_static[eval_mask,1] - ytrue[eval_mask,1]

        # two-layer (ridge a + KF b(t))
        out = run_two_layer_one_id(
            g=g,
            calib_idx=calib_idx,
            lam_ridge=lam_ridge,
            Qb_per_hour_diag=(qb_s, qb_d),
            sigma_R=sigma_R,
            sigma_b0=sigma_b0,
            huber_delta=huber_delta,
            clip_a=clip_a
        )
        y_two = out["y_cal"]
        err_two_s = y_two[eval_mask,0] - ytrue[eval_mask,0]
        err_two_d = y_two[eval_mask,1] - ytrue[eval_mask,1]

        # accumulate micro
        pooled["raw_s"].append(err_raw_s); pooled["raw_d"].append(err_raw_d)
        pooled["static_s"].append(err_sta_s); pooled["static_d"].append(err_sta_d)
        pooled["two_s"].append(err_two_s); pooled["two_d"].append(err_two_d)

        # per-id macro row
        raw_s = me_sd_mae(err_raw_s); raw_d = me_sd_mae(err_raw_d)
        sta_s = me_sd_mae(err_sta_s); sta_d = me_sd_mae(err_sta_d)
        two_s = me_sd_mae(err_two_s); two_d = me_sd_mae(err_two_d)

        per_id_rows.append({
            "id_clean": pid,
            "n_points": int(len(g)),
            "RAW_SBP_ME": raw_s[0], "RAW_SBP_SD": raw_s[1], "RAW_SBP_MAE": raw_s[2],
            "RAW_DBP_ME": raw_d[0], "RAW_DBP_SD": raw_d[1], "RAW_DBP_MAE": raw_d[2],
            "STA_SBP_ME": sta_s[0], "STA_SBP_SD": sta_s[1], "STA_SBP_MAE": sta_s[2],
            "STA_DBP_ME": sta_d[0], "STA_DBP_SD": sta_d[1], "STA_DBP_MAE": sta_d[2],
            "TWO_SBP_ME": two_s[0], "TWO_SBP_SD": two_s[1], "TWO_SBP_MAE": two_s[2],
            "TWO_DBP_ME": two_d[0], "TWO_DBP_SD": two_d[1], "TWO_DBP_MAE": two_d[2],
            "a_s": out["a_s"], "a_d": out["a_d"],
        })

    # micro pooled
    def cat(k): return np.concatenate(pooled[k]) if pooled[k] else np.array([])
    micro = {}
    for tag in ["raw","static","two"]:
        e_s = cat(f"{tag}_s"); e_d = cat(f"{tag}_d")
        m_s = me_sd_mae(e_s);   m_d = me_sd_mae(e_d)
        micro[f"{tag.upper()}_SBP_ME"] = m_s[0]
        micro[f"{tag.upper()}_SBP_SD"] = m_s[1]
        micro[f"{tag.upper()}_SBP_MAE"]= m_s[2]
        micro[f"{tag.upper()}_DBP_ME"] = m_d[0]
        micro[f"{tag.upper()}_DBP_SD"] = m_d[1]
        micro[f"{tag.upper()}_DBP_MAE"]= m_d[2]
    micro["N_EVAL_POINTS"] = int(cat("two_s").shape[0])

    met = pd.DataFrame(per_id_rows)
    macro = met.drop(columns=["id_clean"]).mean(numeric_only=True).to_dict()

    # objective: match your ME±STD preference (lower is better)
    obj = (micro["TWO_SBP_SD"] + micro["TWO_DBP_SD"]) + 1.0 * (abs(micro["TWO_SBP_ME"]) + abs(micro["TWO_DBP_ME"]))
    return micro, macro, obj, met


def grid_search_two_layer(val_csv_path: str, out_csv_path: str, use_tqdm=True):
    lam_list = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    qb_list  = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]  # per hour drift for offsets
    sigma_R_list = [3.0, 4.0, 5.0, 6.0, 8.0]
    sigma_b0_list = [5.0, 10.0, 20.0, 30.0, 50.0]
    huber_list = [15.0, 20.0, 30.0, 40.0]

    combos = list(product(lam_list, qb_list, qb_list, sigma_R_list, sigma_b0_list, huber_list))

    # optional: constrain scale monotonic (often sensible). If you want allow negative slopes, set clip_a=None.
    clip_a = None  # or (0.0, 3.0)

    rows = []
    best = None

    if use_tqdm:
        try:
            from tqdm import tqdm
            it = tqdm(combos, total=len(combos), desc="Two-layer grid", dynamic_ncols=True)
        except Exception:
            it = combos
            use_tqdm = False
    else:
        it = combos

    for step, (lam, qb_s, qb_d, sR, sb0, hd) in enumerate(it, 1):
        micro, macro, obj, _ = eval_params_two_layer_on_val(
            val_csv_path=val_csv_path,
            lam_ridge=lam,
            qb_s=qb_s,
            qb_d=qb_d,
            sigma_R=sR,
            sigma_b0=sb0,
            huber_delta=hd,
            n_awake=4, n_sleep=3, min_gap_min=180,
            clip_a=clip_a
        )

        row = {
            "lam_ridge": lam,
            "qb_s": qb_s, "qb_d": qb_d,
            "sigma_R": sR,
            "sigma_b0": sb0,
            "huber_delta": hd,
            "obj": obj,
            "N_EVAL_POINTS": micro["N_EVAL_POINTS"],
            # compare micro
            "RAW_SBP_SD": micro["RAW_SBP_SD"], "RAW_DBP_SD": micro["RAW_DBP_SD"],
            "STA_SBP_SD": micro["STATIC_SBP_SD"], "STA_DBP_SD": micro["STATIC_DBP_SD"],
            "TWO_SBP_SD": micro["TWO_SBP_SD"], "TWO_DBP_SD": micro["TWO_DBP_SD"],
            "STA_SBP_ME": micro["STATIC_SBP_ME"], "STA_DBP_ME": micro["STATIC_DBP_ME"],
            "TWO_SBP_ME": micro["TWO_SBP_ME"], "TWO_DBP_ME": micro["TWO_DBP_ME"],
            "TWO_SBP_MAE": micro["TWO_SBP_MAE"], "TWO_DBP_MAE": micro["TWO_DBP_MAE"],
        }
        rows.append(row)

        if best is None or obj < best["obj"]:
            best = row

        if use_tqdm:
            it.set_postfix({
                "best_obj": f"{best['obj']:.3f}",
                "best_lam": best["lam_ridge"],
                "best_qb": f"{best['qb_s']}/{best['qb_d']}",
                "best_R": best["sigma_R"],
            })

        # partial save
        if (step % 100) == 0:
            pd.DataFrame(rows).to_csv(out_csv_path.replace(".csv", "_partial.csv"), index=False)

    res = pd.DataFrame(rows).sort_values("obj", ascending=True).reset_index(drop=True)
    res.to_csv(out_csv_path, index=False)
    return res, best

# res, best = grid_search_two_layer("val_set.csv", "two_layer_grid_val.csv", use_tqdm=True)
# print("BEST:", best)
# print(res.head(10))