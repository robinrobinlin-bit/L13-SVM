# src/plotly_example.py
"""
簡易 Plotly Graph Objects 範例，展示如何使用 plotly.graph_objects 建立 3D 散點圖。
此函式可在 Streamlit 中以 ``st.plotly_chart`` 直接呈現。
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def simple_3d_scatter(num_points: int = 100) -> go.Figure:
    """產生隨機 3D 散點圖。

    - 產生 ``num_points`` 個隨機 (x, y, z) 點
    - 使用 Plotly ``graph_objects`` 建立 ``Scatter3d``
    - 回傳 ``go.Figure``，可於 Streamlit 使用 ``st.plotly_chart`` 顯示
    """
    rng = np.random.RandomState(42)
    x = rng.randn(num_points)
    y = rng.randn(num_points)
    z = rng.randn(num_points)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(size=5, color="royalblue", opacity=0.8),
            )
        ]
    )
    fig.update_layout(
        title="簡易 3D 散點圖（plotly.graph_objects）",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig
