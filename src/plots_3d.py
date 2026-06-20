# src/plots_3d.py
"""3D 繪圖模組，使用 plotly 產生互動式 3D Kernel Trick 示意圖。
此模組會將資料點以及模型在高維映射後的決策函數值，以 3D 表面呈現。
"""
import numpy as np
import plotly.graph_objects as go

from .decision_boundary import create_meshgrid, predict_grid


def plot_3d_decision_surface(model, X, y, h=0.1, title="3D Kernel Trick 示意圖"):
    """產生 3D 決策面圖。
    - model: 已訓練的 sklearn SVC
    - X, y: 原始資料
    - h: meshgrid 間距，較大以減少計算量
    回傳 plotly.graph_objects.Figure 物件，可直接於 Streamlit 中 st.plotly_chart 顯示。
    """
    # 建立 2D 網格
    xx, yy = create_meshgrid(X, h=h)
    # 使用模型的決策函數取得距離平面之分數（越正/負代表屬於哪一類）
    if hasattr(model, "decision_function"):
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        zz = model.decision_function(grid_points).reshape(xx.shape)
    else:
        # fallback: 預測類別作為高度
        zz = predict_grid(model, xx, yy)
    # 建立 3D 曲面圖
    fig = go.Figure(data=[
        go.Surface(x=xx, y=yy, z=zz, colorscale="RdBu", opacity=0.7, showscale=False),
        go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=np.zeros_like(X[:, 0]),
            mode="markers",
            marker=dict(size=5, color=y, colorscale="Portland"),
            name="資料點",
        ),
    ])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="特徵 1",
            yaxis_title="特徵 2",
            zaxis_title="決策函數值",
            camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig
