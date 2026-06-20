import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X, y, kernel="linear", C=1.0, gamma="scale", degree=3):
    """Train an SVM model with given hyper‑parameters.

    Parameters
    ----------
    X, y : array‑like
        Training data and labels.
    kernel : str, optional
        "linear", "rbf" or "poly".
    C : float, optional
        Regularisation parameter.
    gamma : "scale", "auto" or float, optional
        Kernel coefficient for ``rbf`` and ``poly``.
    degree : int, optional
        Degree for ``poly`` kernel.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=False)
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    return model, acc
