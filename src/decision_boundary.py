# -*- coding: utf-8 -*-
"""src/decision_boundary.py
計算決策面格點，用於 Plotly 繪圖。
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator


def make_meshgrid(x: np.ndarray, y: np.ndarray, h: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """在特徵範圍內建立網格座標。

    Parameters
    ----------
    x, y : np.ndarray
        原始特徵向量（1D）
    h : float, optional
        網格間距，預設 0.02。
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def predict_grid(model: BaseEstimator, xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
    """對網格座標進行模型預測，返回類別編號的 2D 陣列。"""
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    return Z.reshape(xx.shape)
