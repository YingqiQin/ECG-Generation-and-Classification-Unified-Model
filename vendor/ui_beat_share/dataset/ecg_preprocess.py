import scipy
from scipy.signal import (
    convolve, butter,
    lfilter, resample,
    hilbert, filtfilt, medfilt,
    savgol_filter, iirnotch
)
import scipy.signal as signal
from scipy.signal import find_peaks
import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
import pywt
import os

def pp(data):
    x = np.max(abs(data))
    if x > 10:
        b = np.argwhere(abs(data) > 10)
        for k in b[:,0]:
            if k > 0:
                data[k] = data[k-1]
    return data

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def highpass_filter(signal, hpf, fs):

    hp = hpf / (fs*0.5)
    hpb, hpa = butter(5, hp, 'highpass')
    hp_signal = filtfilt(hpb, hpa, signal, axis=0)

    return hp_signal

def lowpass_filter(signal, lpf, fs):

    lp = lpf / (fs*0.5)
    lpb, lpa = butter(5, lp, 'lowpass')
    lp_signal = filtfilt(lpb, lpa, signal, axis=0)

    return lp_signal

def notch_filter(signal, ntf):

    nt = ntf / nyq
    ntb, nta = iirnotch(nt, 30)
    nt_signal = filtfilt(ntb, nta, signal, axis=0)

    return nt_signal

def remove_outliers(signal, t):

    signalc = np.copy(signal)
    std = np.std(signalc)
    t_std = t * std
    outliers = np.where(np.abs(signalc) > t_std)
    signalc[outliers] = t_std

    return signalc

def enhancement(signal, swin, lwin):

    signal = remove_outliers(signal, 3)
    signal2 = np.power(signal, 2)

    swin_filter = np.ones((swin, 1))
    lwin_filter = np.ones((lwin, 1))

    signal2_sf = convolve(signal2, swin_filter, 'same', 'auto')
    signal2_lf = convolve(signal2, lwin_filter, 'same', 'auto')

    coeff = signal2_sf / signal2_lf
    enhanced_signal = coeff * signal

    return enhanced_signal

def median_filter(signal, size=5):

    med_signal = medfilt(signal.flatten(), size)
    return med_signal

# Wiener filter function
def wiener(sig, mysize=None, noise=None):
    sig = np.asarray(sig)

    if mysize is None:
        mysize = [3] * len(sig)
    mysize = np.asarray(mysize)
    if mysize.shape == ():
        mysize = np.repeat(mysize.item(), sig.ndim)

    # Estimate the local mean
    lMean = scipy.signal.correlate(sig, np.ones(mysize), 'same') / np.product(mysize, axis=0)

    # Estimate the local variance
    lVar = (scipy.signal.correlate(sig ** 2, np.ones(mysize), 'same') / np.product(mysize, axis=0) - lMean ** 2)

    # Estimate the noise power if needed.
    if noise is None:
        noise = np.mean(np.ravel(lVar), axis=0)

    res = sig - lMean
    res *= (1 - noise / (lVar+0.0001))
    res += lMean
    out = np.where(lVar < noise, lMean, res)

    return out

def local_wiener(sig, mysize=10, local_size=21, fs=2000):
    sig = np.asarray(sig)

    pn = np.var(np.reshape(sig, (-1, local_size)), axis=-1)
    pn_med = np.quantile(pn, 0.25)
    pn_min = np.min(pn)
    pn_m = (pn_med + pn_min) / 2

    sig_filted = wiener(sig, mysize, noise=pn_m)

    return sig_filted

# homomorphic envelope with hilbert
def homomorphic_envelope(x, fs=1000, f_LPF=8, order=3):
    """
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b, a = butter(order, 2 * f_LPF / fs, 'low')
    he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(x)))))
    he[0] = he[1]
    return he

# Normalization Function
def temporal_norm(input):
    x = preprocessing.minmax_scale(input)
    return x

# Normalization Function
def norm(input):
    x = preprocessing.scale(input)
    return x

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def downsample(data, rate, new_rate):
    num = int(len(data) * new_rate / rate)
    y = scipy.signal.resample(data, num)
    return y

def feature_extract(ecg):
    b = [0.0564484622607365, 0.112896924521473, 0.0564484622607365]
    a = [1,-1.22465158101310, 0.450445430056041]
    ecg = signal.filtfilt(b, a, ecg, method='pad')

    m = signal.medfilt(ecg, 251)
    m = signal.medfilt(m, 251)

    E = ecg - m

    max_ = np.max(E)
    min_ = np.min(E)

    la = -1
    lb = 1
    k = (lb-la) / (max_-min_)
    ecg = la + k * (ecg - min_)

    ecg_feature = np.transpose(np.array([ecg]))
    ecg_feature = temporal_norm(ecg_feature)

    return ecg_feature


