"""Compute raw-scale and normalized RUL regression metrics."""
import numpy as np

def metrics(y, p):
    """Evaluate predictions using the actual test-label maximum and range."""
    y, p = np.asarray(y), np.asarray(p)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError('Invalid prediction')
    raw = float(np.sqrt(np.mean((y-p)**2)))
    return dict(rmse_raw=raw, rmse_normalized=raw/float(y.max()),
                nrmse_range=raw/float(np.ptp(y)), mae=float(np.mean(abs(y-p))),
                r2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)))
