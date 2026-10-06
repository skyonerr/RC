"""Load bearing features and construct windows for offline trajectory evaluation."""
import pickle
import numpy as np

def create_sliding_windows(data_2d, window_size):
    """Construct temporal feature windows from a bearing trajectory."""
    n_time, n_feat = data_2d.shape
    if n_time <= window_size: return np.tile(data_2d, (1, 1, 1))
    shape = (n_time - window_size + 1, window_size, n_feat)
    strides = (data_2d.strides[0], data_2d.strides[0], data_2d.strides[1])
    return np.lib.stride_tricks.as_strided(data_2d, shape=shape, strides=strides)


def get_global_stats(train_pkls):
    """Fit shared feature statistics from the selected training prefixes."""
    X_list = []
    for pkl in train_pkls:
        with open(pkl, "rb") as f: data = pickle.load(f)
        n = int(len(data["train"]["X_vib"]) * 0.50)
        X_list.append(data["train"]["X_vib"][:max(n, 100)])
    X_cat = np.concatenate(X_list)
    return np.mean(X_cat, axis=0), np.std(X_cat, axis=0) + 1e-9


def load_data(pkl_path, mean, std, win=25):
    """Append full-trajectory position and align windows with endpoint labels."""
    with open(pkl_path, "rb") as f: data = pickle.load(f)
    X = (data["train"]["X_vib"] - mean) / std
    # This offline coordinate requires the complete retained trajectory length.
    time = np.linspace(0, 1, len(X)).reshape(-1, 1)
    X = np.concatenate([X, time], axis=1)
    X_win = create_sliding_windows(X, win)

    y_raw = data["train"]["y_10s"][win - 1:]
    T = data["train"]["T_K"][win - 1:]
    y = np.asarray(y_raw, dtype=float)

    min_len = min(len(X_win), len(y), len(T))
    return X_win[:min_len], T[:min_len], y[:min_len]


