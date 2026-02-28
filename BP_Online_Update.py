import pandas as pd

def add_day_column(df: pd.DataFrame, tz="America/New_York", assume_utc=True):
    """
    t_bp_ms: epoch milliseconds
    assume_utc=True 表示 t_bp_ms 是 UTC epoch ms（最常见）
    """
    ts = pd.to_datetime(df["t_bp_ms"], unit="ms", utc=True) if assume_utc else pd.to_datetime(df["t_bp_ms"], unit="ms")
    if assume_utc:
        ts = ts.dt.tz_convert(tz)
    df = df.copy()
    df["dt_local"] = ts
    df["day"] = df["dt_local"].dt.date.astype(str)  # 'YYYY-MM-DD'
    df["ts_sec"] = (df["t_bp_ms"].astype("int64") // 1000).astype("int64")
    return df

import numpy as np

class BayesianAffine1D:
    """
    y = a*yhat + b + eps,  theta=[a,b] ~ N(mu, Sigma)
    """
    def __init__(self, mu=None, Sigma=None, sigma_obs=5.0):
        self.mu = np.array([1.0, 0.0], float) if mu is None else np.asarray(mu, float).copy()
        self.Sigma = np.diag([1.0**2, 30.0**2]).astype(float) if Sigma is None else np.asarray(Sigma, float).copy()
        self.sigma2 = float(sigma_obs)**2

    @staticmethod
    def x(yhat):
        return np.array([float(yhat), 1.0], float)

    def predict(self, yhat):
        x = self.x(yhat)
        mean = float(x @ self.mu)
        var  = float(x @ self.Sigma @ x + self.sigma2)
        return mean, np.sqrt(max(var, 1e-12))

    def update(self, yhat, y_true):
        x = self.x(yhat)
        y_true = float(y_true)
        r = y_true - float(x @ self.mu)
        s = float(self.sigma2 + x @ self.Sigma @ x)
        K = (self.Sigma @ x) / max(s, 1e-12)
        self.mu = self.mu + K * r
        self.Sigma = self.Sigma - np.outer(K, x) @ self.Sigma

    def inflate(self, gamma=0.0):
        # 跨天/长时间无 cuff：轻微增大不确定性，防过度自信
        if gamma > 0:
            self.Sigma = self.Sigma + gamma * np.eye(2)