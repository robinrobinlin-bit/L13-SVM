import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles, make_classification

def generate_dataset(name: str, n_samples: int = 200, noise: float = 0.1, random_state: int = 42):
    """Generate a synthetic dataset.

    Parameters
    ----------
    name: str
        One of "blobs", "moons", "circles", "classification".
    n_samples: int
        Number of total samples.
    noise: float
        Noise level for "moons" and "circles".
    random_state: int
        Seed for reproducibility.
    """
    if name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=2, random_state=random_state)
    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif name == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported dataset name: {name}")
    return X, y
