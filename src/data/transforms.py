import numpy as np
from scipy.signal import butter, filtfilt

def forward_fill_impute(data):
    """
    Forward-fill missing values in sensor data.
    """
    return np.where(np.isnan(data), np.roll(data, 1, axis=0), data)

def low_pass_filter(data, cutoff, fs, order=4):
    """
    Apply low-pass filter to sensor data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)

def normalize(data, mean, std):
    """
    Normalize data using mean and std.
    """
    return (data - mean) / std
