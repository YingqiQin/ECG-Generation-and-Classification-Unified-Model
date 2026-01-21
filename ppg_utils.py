import numpy as np
from scipy.signal import savgol_filter

def ppg_to_vpg_apg(ppg_1d: np.ndarray, fs: int = 100,
                   win_sec: float = 0.21, poly: int = 3):
    """
    ppg_1d: shape [T]
    return: x0,x1,x2 each shape [T]
    """
    # window length (odd)
    win = int(round(win_sec * fs))
    win = max(win, poly + 2)
    if win % 2 == 0:
        win += 1

    # 0th: smoothed ppg
    x0 = savgol_filter(ppg_1d, window_length=win, polyorder=poly, deriv=0, mode="interp")
    # 1st deriv (VPG), units per second
    x1 = savgol_filter(ppg_1d, window_length=win, polyorder=poly, deriv=1, delta=1/fs, mode="interp")
    # 2nd deriv (APG)
    x2 = savgol_filter(ppg_1d, window_length=win, polyorder=poly, deriv=2, delta=1/fs, mode="interp")

    # channel-wise robust scaling / z-score（建议至少做其一）
    def z(x):
        m = np.mean(x); s = np.std(x) + 1e-6
        return (x - m) / s

    return z(x0), z(x1), z(x2)

