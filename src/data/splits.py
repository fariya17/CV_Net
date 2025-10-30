import numpy as np
from sklearn.model_selection import KFold

def train_val_test_split(X, y, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Split data into train, validation, and test sets.
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return X[train_idx], X[val_idx], X[test_idx], y[train_idx], y[val_idx], y[test_idx]

def kfold_split(n_samples, n_folds=5, seed=42):
    """
    Generate k-fold indices.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(kf.split(np.arange(n_samples)))
