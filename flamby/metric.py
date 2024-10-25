import numpy as np
from sklearn import metrics

def metric_FHD(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        return ((y_pred > 0.0) == y_true).mean()
    except ValueError:
        return np.nan

def metric_FISIC(y_true, y_pred):
    y_true = y_true.reshape(-1)
    return metrics.balanced_accuracy_score(y_true, y_pred)