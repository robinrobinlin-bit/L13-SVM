# src/plots_3d.py
"""
3D 繪圖模組 – 用於展示 Kernel Trick 概念 (將 2D 資料映射到 3D 空間)

核心映射函式為: z = x^2 + y^2
此模組使用 Plotly 產生可旋轉、縮放的 3D 散點圖，並在圖中加入概念性決策平面。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def kernel_mapping_3d(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """將 2D 資料 X 映射到 3D 空間，z = x^2 + y^2。

    參數說明:
    - X: shape (n_samples, 2) 的特徵矩陣
    - y: shape (n_samples,) 的標籤 (0 或 1)
    
    回傳:
    - Plotly Figure，包含 3D 散點與簡易的概念決策平面 (z = x^2 + y^2)。
    """
    # 計算 z 座標
    z = np.square(X[:, 0]) + np.square(X[:, 1])

    # 建立顏色對應
    colors = np.where(y == 0, "royalblue", "orange")
    labels = np.where(y == 0, "Class 0", "Class 1")

    fig = go.Figure()
    # 3D scatter
    fig.add_trace(
        go.Scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=z,
            mode="markers",
            marker=dict(size=4, color=colors),
            name="Data Points",
            hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>",
        )
    )
    # 加一個概念平面 (實際上是 z = x^2 + y^2 的曲面) – 使用 meshgrid 繪製曲面
    grid_x = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 30)
    grid_y = np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 30)
    gx, gy = np.meshgrid(grid_x, grid_y)
    gz = np.square(gx) + np.square(gy)
    fig.add_trace(
        go.Surface(
            x=gx,
            y=gy,
            z=gz,
            colorscale="Greys",
            opacity=0.4,
            name="Mapping Surface",
            showscale=False,
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z = x² + y²",
        ),
        title="Kernel Trick 概念 – 2D → 3D 映射",
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def plot_3d_decision_surface_concept(X: np.ndarray, y: np.ndarray) -> go.Figure:
    """展示使用 RBF kernel 產生的非線性決策面概念（示意圖）。
    
    此函式先以 kernel_mapping_3d 繪製映射曲面，然後在同一圖中加入一條概念決策平面。
    決策平面僅為示意，使用 z = 0 作為分割平面（實際 RBF 會產生更複雜的曲面）。
    """
    fig = kernel_mapping_3d(X, y)
    # 添加概念決策平面 (z = constant) 以示範在 3D 中的線性分割
    # 這裡選擇 z = np.mean(z) 作為平面高度
    z_mean = np.mean(np.square(X[:, 0]) + np.square(X[:, 1]))
    # 建立平面座標
    plane_x = np.array([X[:, 0].min() - 0.5, X[:, 0].max() + 0.5])
    plane_y = np.array([X[:, 1].min() - 0.5, X[:, 1].max() + 0.5])
    px, py = np.meshgrid(plane_x, plane_y)
    pz = np.full_like(px, z_mean)
    fig.add_trace(
        go.Surface(
            x=px,
            y=py,
            z=pz,
            colorscale=[[0, "rgba(0,255,0,0.2)"], [1, "rgba(0,255,0,0.2)"]],
            showscale=False,
            name="Decision Plane (概念)",
            opacity=0.5,
        )
    )
    fig.update_layout(title="Kernel Trick – 3D Conceptual Decision Surface")
    return fig
