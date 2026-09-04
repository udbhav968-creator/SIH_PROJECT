"""
ROAD-SHIELD Mini-Batch DataLoader & Train/Val Splitter
"""
import numpy as np

def train_val_split(*arrays, val_ratio=0.2, seed=42):
    np.random.seed(seed)
    n = len(arrays[0])
    indices = np.random.permutation(n)
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    result = []
    for arr in arrays:
        result.append(arr[train_idx])
        result.append(arr[val_idx])
    return tuple(result)

def get_batches(X, y, batch_size=64, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, n, batch_size):
        end_idx = min(start_idx + batch_size, n)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]
