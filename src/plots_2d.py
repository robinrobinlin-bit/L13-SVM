# src/plots_2d.py
"""
2D 繪圖模組 – 用於顯示 SVM 的決策邊界、support vectors 以及 margin。

此模組使用 Matplotlib，返回 ``matplotlib.figure.Figure`` 物件，
在 Streamlit 中可透過 ``st.pyplot`` 直接呈現。
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from .decision_boundary import create_meshgrid, predict_decision_boundary, get_decision_function


def plot_decision_boundary_2d(
    X: np.ndarray,
    y: np.ndarray,
    model: SVC,
    resolution: int = 100,
    show_margin: bool = True,
) -> plt.Figure:
    """繪製 2D 決策邊界與支援向量。

    參數說明:
    - X, y: 原始資料
    - model: 已訓練好的 sklearn SVC
    - resolution: meshgrid 解析度，預設 100（對應 100×100 網格）
    - show_margin: 是否在圖中顯示 margin 等高線（decision_function = +/-1）
    """
    # 建立格點
    xx, yy = create_meshgrid(X, resolution=resolution)
    # 類別預測
    Z = predict_decision_boundary(model, xx, yy)
    # 決策函數（用於 margin）
    Z_score = get_decision_function(model, xx, yy) if show_margin else None

    fig, ax = plt.subplots(figsize=(6, 5))
    # 決策邊界顏色映射
    cmap = plt.cm.RdYlBu
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=np.arange(-0.5, 2, 1))
    # 原始點
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor="k", s=40)
    # 支援向量
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100, facecolors="none", edgecolors="gold", linewidths=2, label="Support Vectors")
    # Margin 等高線（若有）
    if Z_score is not None:
        ax.contour(xx, yy, Z_score, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("SVM Decision Boundary (2D)")
    ax.legend()
    return fig
