# -*- coding: utf-8 -*-
"""src/plots_3d.py
使用 Plotly 繪製 3D 決策面（適用 RBF、Poly 核心）。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from .decision_boundary import make_meshgrid, predict_grid


def plot_decision_boundary_3d(model, X: np.ndarray, y: np.ndarray, title: str = "3D Decision Boundary"):
    """回傳 Plotly Figure，顯示 3D 數據與決策面。
    只支援 3 維特徵（此範例使用前兩維繪製平面，第三維作為顏色或大小）
    """
    # 只取前兩維繪製平面，第三維用於點的顏色
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    Z = predict_grid(model, xx, yy)

    fig = go.Figure()
    # 決策面等高線（投影在 xy 平面）
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
    # 數據點（使用第三維作為顏色）
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=X[:, 2] if X.shape[1] > 2 else np.zeros_like(X[:, 0]),
            mode="markers",
            marker=dict(
                size=5,
                color=y,
                colorscale=[[0, "rgb(31,119,180)"], [1, "rgb(255,127,14)"]],
                opacity=0.8,
            ),
            name="資料點",
        )
    )
    # 支援向量（若有）
    if hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        fig.add_trace(
            go.Scatter3d(
                x=sv[:, 0],
                y=sv[:, 1],
                z=sv[:, 2] if sv.shape[1] > 2 else np.zeros_like(sv[:, 0]),
                mode="markers",
                marker=dict(size=8, color="gold", symbol="diamond"),
                name="支援向量",
            )
        )
    fig.update_layout(title=title, scene=dict(xaxis_title="X1", yaxis_title="X2", zaxis_title="X3"))
    return fig
