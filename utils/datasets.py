# src/data_generator.py
"""
Data generation utilities for the SVM 教學專案。

提供四種常見的二元分類資料集，支援使用者自行指定樣本數（100~500）
以及隨機噪聲程度。所有資料皆返回 NumPy 陣列 (X, y)。
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs


def _validate_n_samples(n_samples: int) -> int:
    """驗證樣本數是否在 100~500 範圍內，若超出則自動截取。

    Returns the (clamped) integer number of samples.
    """
    if n_samples < 100:
        return 100
    if n_samples > 500:
        return 500
    return n_samples


def generate_linear_data(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """產生線性可分的二維資料。

    參數說明:
    - n_samples: 樣本筆數，會被限制在 100~500 之間。
    - noise: 高斯噪聲的標準差，預設 0.1。
    - random_state: 讓結果可重現。
    """
    n_samples = _validate_n_samples(n_samples)
    rng = np.random.RandomState(random_state)
    # 隨機產生點，之後依據 x0 + x1 > 0 分類
    X = rng.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += noise * rng.randn(*X.shape)
    return X, y


def generate_moons_data(
    n_samples: int = 200,
    noise: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """產生 classic "moons" 資料集（兩個半月形），使用 sklearn 的 make_moons。"""
    n_samples = _validate_n_samples(n_samples)
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def generate_circles_data(
    n_samples: int = 200,
    noise: float = 0.1,
    factor: float = 0.5,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """產生 concentric circles（同心圓）資料集。"""
    n_samples = _validate_n_samples(n_samples)
    X, y = make_circles(
        n_samples=n_samples,
        noise=noise,
        factor=factor,
        random_state=random_state,
    )
    return X, y


def generate_blobs_data(
    n_samples: int = 200,
    centers: int = 2,
    cluster_std: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """產生高斯混合 (blobs) 資料集。

    參數 `centers` 目前固定為 2，以符合二元分類需求。
    """
    n_samples = _validate_n_samples(n_samples)
    result = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    X = result[0]
    y = result[1]
    if noise > 0:
        rng = np.random.RandomState(random_state)
        X += noise * rng.randn(*X.shape)
    return X, y


def generate_dataset(name: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """根據名稱分派對應的資料產生函式。"""
    dispatch = {
        "linear": generate_linear_data,
        "moons": generate_moons_data,
        "circles": generate_circles_data,
        "blobs": generate_blobs_data,
    }
    if name not in dispatch:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(dispatch.keys())}")
    return dispatch[name](**kwargs)


# ------------------------------------------------------------
# 範例使用（在 Jupyter 或 Streamlit 中可直接呼叫）
# ------------------------------------------------------------
# ```python
# X, y = generate_linear_data(n_samples=300, noise=0.05)
# X, y = generate_dataset("moons", n_samples=300)
# ```
