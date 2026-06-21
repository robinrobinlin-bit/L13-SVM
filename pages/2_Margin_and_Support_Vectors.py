# -*- coding: utf-8 -*-
"""pages/2_Margin_and_Support_Vectors.py
說明 SVM 的 margin 與支援向量概念，並以互動圖表展示。
"""

import streamlit as st
import numpy as np
from pathlib import Path

from src.data_generator import make_dataset
from src.svm_model import get_model
from src.plots_2d import plot_decision_boundary_2d


def run():
    st.title("2️⃣ Margin & Support Vectors")
    st.markdown(
        """
        **Margin** 為決策平面與最近支援向量之間的距離，
        越大代表模型對樣本的容錯性越好。
        **Support Vectors** 為位於 margin 內或正好在決策平面上的資料點，
        它們決定了模型的形狀。
        """
    )

    # 側欄參數設定
    st.sidebar.header("🛠 參數設定")
    dataset_name = st.sidebar.selectbox("資料集", ["blobs", "moons", "circles", "classification"], index=0)
    n_samples = st.sidebar.slider("樣本數量", 100, 500, 250, step=50)
    noise = st.sidebar.slider("噪聲", 0.0, 0.5, 0.1, step=0.05)
    C = st.sidebar.slider("C (正則化)", 0.01, 10.0, 1.0, step=0.01)
    kernel = "linear"

    # 產生資料
    X, y, _ = make_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=42)

    # 訓練模型
    model, accuracy = get_model(X, y, kernel=kernel, C=C, gamma="scale", degree=3)

    # 繪圖
    fig = plot_decision_boundary_2d(model, X, y, title="Margin & Support Vectors")
    st.plotly_chart(fig, use_container_width=True)

    # 模型資訊
    st.subheader("模型資訊")
    st.write(f"**支援向量數量**: {len(model.support_vectors_)} / {len(y)}")
    st.write(f"**訓練準確度**: {accuracy * 100:.2f}%")
    if hasattr(model, "coef_"):
        w = model.coef_[0]
        margin = 1 / np.linalg.norm(w)
        st.write(f"**Margin 距離**: {margin:.3f}")
        st.latex(r"\mathbf{w} \cdot \mathbf{x} + b = 0")
        st.latex(r"\text{margin}=\frac{2}{\|\mathbf{w}\|}")
        # Support Vectors intro video
        video_path = Path(__file__).parent.parent / "assets" / "videos" / "support_vectors_intro.mp4"
        if video_path.is_file():
            st.video(str(video_path))
        else:
            st.info(f"Video not found: {video_path.name}")

run()