import numpy as np

def create_sliding_windows(data, window_size, overlap):
    """
    Create sliding windows with specified overlap.
    """
    step = int(window_size * (1 - overlap))
    windows = []
    for i in range(0, len(data) - window_size + 1, step):
        windows.append(data[i:i + window_size])
    return np.array(windows)
