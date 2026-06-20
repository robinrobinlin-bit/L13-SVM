# pages/1_Linear_SVM.py
"""
線性 SVM 教學頁面 – 展示 Linear Kernel 的決策線、margin、support vectors。
"""
import streamlit as st
import numpy as np

from utils.datasets import generate_dataset
from utils.svm_model import train_svm_model
from utils.plotting import plot_decision_boundary

st.title("1️⃣ Linear SVM")

st.markdown(
    """
    **Linear SVM** 使用線性核 (kernel="linear")，在特徵空間中以直線（2D）或超平面（高維）分割資料。
    - **Margin** 為決策線與最近的 support vectors 之間的距離，
    - **Support Vectors** 為位於 margin 上的資料點，決定最終模型。
    """
)

# 側欄參數
st.sidebar.header("🛠 參數設定")

# 資料集
 dataset_name = st.sidebar.selectbox(
    "Dataset (資料集)",
    ["linear", "moons", "circles"],
    index=0,
 )
 n_samples = st.sidebar.slider("樣本數量", 100, 500, 250, step=50)
 noise = st.sidebar.slider("噪聲", 0.0, 0.5, 0.1, step=0.05)
 C = st.sidebar.slider("C (正則化)", 0.01, 100.0, 1.0, step=0.01)

# 產生資料
 X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=42)

# 訓練 Linear SVM
 model = train_svm_model(X, y, kernel="linear", C=C)

# 繪圖顯示決策線、margin、支援向量
 fig = plot_decision_boundary(model, X, y, title="Linear SVM Decision Boundary")
 st.plotly_chart(fig, use_container_width=True)

# 模型資訊
st.subheader("模型資訊")
st.write(f"**支援向量數量**: {len(model.support_vectors_)} / {len(y)}")
st.write(f"**訓練準確度**: {model.score(X, y) * 100:.2f}%")

# margin 計算與公式說明
w = model.coef_[0]
b = model.intercept_[0]
margin = 1 / np.linalg.norm(w)
st.write(f"**Margin 距離**: {margin:.3f}")
st.latex(r"\mathbf{w} \cdot \mathbf{x} + b = 0")
st.latex(r"\text{margin}=\frac{2}{\|\mathbf{w}\|}")
