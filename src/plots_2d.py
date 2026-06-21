# -*- coding: utf-8 -*-
"""src/plots_2d.py
使用 Plotly 繪製 2D 決策邊界。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from .decision_boundary import make_meshgrid, predict_grid


def plot_decision_boundary_2d(model, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary"):
    """回傳 Plotly Figure，顯示資料點、支援向量與決策面。

    Parameters
    ----------
    model : sklearn estimator
        已訓練好的 SVC 模型。
    X, y : np.ndarray
        資料點與標籤。
    title : str, optional
        圖表標題。
    """
    # 建立網格
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    Z = predict_grid(model, xx, yy)

    # 顏色對應
    colors = np.where(y == 0, "rgba(31, 119, 180,0.5)", "rgba(255, 127, 14,0.5)")

    fig = go.Figure()
    # 決策面色塊
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=Z,
            showscale=False,
            colorscale=[[0, "rgb(31,119,180)"], [1, "rgb(255,127,14)"]],
            opacity=0.3,
            hoverinfo="skip",
        )
    )
    # 資料點
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(color=colors, size=8, line=dict(width=1, color="DarkSlateGrey")),
            name="資料點",
        )
    )
    # 支援向量（如果模型有此屬性）
    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        fig.add_trace(
            go.Scatter(
                x=sv[:, 0],
                y=sv[:, 1],
                mode="markers",
                marker=dict(color="gold", size=12, symbol="star"),
                name="支援向量",
            )
        )
    fig.update_layout(title=title, xaxis_title="X1", yaxis_title="X2", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig
