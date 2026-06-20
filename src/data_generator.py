# src/data_generator.py
"""資料產生模組，提供各種合成資料集供 SVM 教學使用。
使用 sklearn.datasets 產生 linear、moons、circles、blobs 四種資料。
所有函式回傳 (X, y) 其中 X 為 shape (n_samples, 2) 的特徵陣列，y 為標籤。
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


def generate_linear_data(n_samples=200, noise=0.1, random_state=42):
    """產生線性可分的二元資料。
    透過兩個高斯中心點產生，加入少量 noise。
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=2,
        n_features=2,
        cluster_std=0.8,
        random_state=random_state,
    )
    # 加入噪聲
    X += noise * np.random.randn(*X.shape)
    return X, y


def generate_moons_data(n_samples=200, noise=0.2, random_state=42):
    """產生兩月形（moons）資料，非線性可分的典型例子。"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y


def generate_circles_data(n_samples=200, noise=0.2, factor=0.5, random_state=42):
    """產生同心圓資料，適合展示 kernel trick。"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y


def generate_blobs_data(n_samples=200, centers=3, cluster_std=1.0, random_state=42):
    """產生多中心的 blob 資料，用於展示多類別或較雜訊的情境。"""
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    return X, y
