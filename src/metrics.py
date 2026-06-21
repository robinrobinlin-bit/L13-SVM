from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from typing import Dict, Any

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Calculate common classification metrics.

    Returns a dictionary with keys:
    - ``accuracy``
    - ``precision``
    - ``recall``
    - ``f1``
    - ``confusion`` (2x2 numpy array)
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred),
    }
    return metrics
