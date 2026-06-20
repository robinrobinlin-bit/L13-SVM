# src/plots_2d.py
"""2D 繪圖模組，使用 matplotlib 產生決策邊界圖，並回傳 figure 物件，便於在 Streamlit 中顯示。"""
import matplotlib.pyplot as plt
import numpy as np

from .decision_boundary import create_meshgrid, predict_grid


def plot_2d_decision_boundary(model, X, y, h=0.05, title="2D 決策邊界"):
    """繪製 2D 決策邊界與訓練資料點。
    - model: 已訓練的 sklearn SVC
    - X, y: 原始資料
    - h: meshgrid 間距
    - title: 圖表標題
    會回傳 matplotlib.figure.Figure 物件。
    """
    xx, yy = create_meshgrid(X, h=h)
    Z = predict_grid(model, xx, yy)

    fig, ax = plt.subplots(figsize=(6, 5))
    # 決策區域顏色
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.6)
    # 訓練點
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    # support vectors
    if hasattr(model, "support_vectors_"):
        ax.scatter(
            model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=100,
            linewidth=1,
            facecolors="none",
            edgecolors="yellow",
            label="Support Vectors",
        )
    ax.set_xlabel("特徵 1")
    ax.set_ylabel("特徵 2")
    ax.set_title(title)
    ax.legend()
    return fig
