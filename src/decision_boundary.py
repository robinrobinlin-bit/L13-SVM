# src/decision_boundary.py
"""決策邊界計算模組。
提供 `create_meshgrid` 產生測試座標格點，以及 `predict_grid` 用模型在格點上預測，返回 Z 用於繪圖。"""
import numpy as np


def create_meshgrid(X, h=0.05, x_limits=None, y_limits=None):
    """建立 2D 網格座標。
    - X: 原始特徵陣列 (用於自動推算範圍)
    - h: 格點間距，預設 0.05
    - x_limits/y_limits: 手動指定範圍 (tuple(min, max))，若未提供則根據 X 計算 ±1.5 倍範圍。
    回傳 (xx, yy) meshgrid arrays。
    """
    if x_limits is None:
        x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    else:
        x_min, x_max = x_limits
    if y_limits is None:
        y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    else:
        y_min, y_max = y_limits
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def predict_grid(model, xx, yy):
    """在 meshgrid 上使用已訓練的 SVM model 進行預測。
    - model: 已訓練好的 sklearn SVC
    - xx, yy: meshgrid arrays
    回傳 Z, shape 與 xx, yy 相同，值為預測 class。"""
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    return Z
