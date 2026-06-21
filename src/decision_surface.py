import numpy as np
from typing import Tuple

def mesh_grid(X: np.ndarray, resolution: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a mesh grid covering the range of ``X``.

    Parameters
    ----------
    X: np.ndarray, shape (n_samples, 2)
        Input data points.
    resolution: int, default 150
        Number of points per axis (max 150 to limit memory).
    Returns
    -------
    xx, yy: np.ndarray
        Meshgrid arrays ready for contour plotting.
    """
    # Clip resolution to safe upper bound
    resolution = min(resolution, 150)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    return xx, yy

def predict_surface(model, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """Compute model predictions over a mesh grid.

    The function reshapes the grid into ``(n_points, 2)`` before calling the
    model's ``predict`` method and then reshapes the result back to the grid
    shape.
    """
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    return Z.reshape(xx.shape)
