# pages/1_Visualization.py
"""
SVM 可視化互動頁面
展示 Linear、RBF、Polynomial 三種 kernel 的決策邊界、支援向量、margin（線性）
使用者可以調整 C、gamma、degree 等參數即時觀察效果。
"""

import streamlit as st
import numpy as np
from pathlib import Path

# 專案內部模組
from utils.datasets import generate_dataset
from utils.svm_model import train_svm_model
from utils.plotting import plot_decision_boundary

st.title("📈 SVM 決策邊界視覺化")

st.markdown(
    """
    本頁面提供 **Linear SVM**、**RBF SVM** 以及 **Polynomial SVM** 三種核函式的即時可視化。
    - **Linear**：顯示決策線、正負 margin、支援向量。
    - **RBF**：非線性曲面（使用 2D 投影展示）。
    - **Polynomial**：多項式曲線邊界。
    請使用左側側欄調整參數，即時觀察模型變化。
    """
)

st.sidebar.header("🛠 參數設定")

# 資料集選擇（提供線性可分與非線性示例）
 dataset_name = st.sidebar.selectbox(
    "Dataset (資料集)",
    ["linear", "moons", "circles"],
    index=0,
)

# 共有樣本數與噪聲控制
 n_samples = st.sidebar.slider("樣本數量", 100, 500, 250, step=50)
 noise = st.sidebar.slider("噪聲", 0.0, 0.5, 0.1, step=0.05)

# SVM 參數
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=0)
C = st.sidebar.slider("C (正則化)", 0.01, 100.0, 1.0, step=0.01)
if kernel in ["rbf", "poly", "sigmoid"]:
    gamma = st.sidebar.slider("Gamma", 0.001, 10.0, 1.0, step=0.001)
    # gamma 說明：
    #   - 小值 → RBF 核的影響範圍較大，決策邊界較平滑。
    #   - 大值 → 核函式變窄，模型會過度擬合資料，決策邊界變得扭曲。
    st.caption("*γ（Gamma） 越小，模型較平滑；γ 越大，模型容易過擬合，呈現較複雜的曲線。*")
else:
    gamma = "scale"
if kernel == "poly":
    degree = st.sidebar.slider("Degree", 2, 5, 3, step=1)
else:
    degree = 3

# 產生資料
X, y = generate_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=42)

# 訓練模型
model = train_svm_model(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)

# 繪圖
fig = plot_decision_boundary(model, X, y, title=f"Kernel: {kernel.upper()}", resolution=100)
st.plotly_chart(fig, use_container_width=True)

# 顯示模型資訊與公式說明
st.subheader("模型資訊")
st.write(f"**支援向量數量**: {len(model.support_vectors_)} / {len(y)}")
st.write(f"**訓練準確度**: {model.score(X, y) * 100:.2f}%")
if kernel == "linear":
    # margin 計算（已在圖中顯示）
    w = model.coef_[0]
    b = model.intercept_[0]
    margin = 1 / np.linalg.norm(w)
    st.write(f"**Margin 距離**: {margin:.3f}")

# 公式與解釋
st.subheader("公式與解釋")
if kernel == "linear":
    st.latex(r"\mathbf{w} \cdot \mathbf{x} + b = 0")
    st.latex(r"\text{margin}=\frac{2}{\|\mathbf{w}\|}")
elif kernel == "rbf":
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x}-\mathbf{x}'\|^2)")
elif kernel == "poly":
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = (\gamma \mathbf{x}^\top \mathbf{x}' + r)^{d}")
else:  # sigmoid
    st.latex(r"K(\mathbf{x}, \mathbf{x}') = \tanh(\gamma \mathbf{x}^\top \mathbf{x}' + r)")

# 嵌入 Manim 動畫（若檔案存在）
if st.checkbox("顯示 Manim 動畫 (僅本機)"):
    video_path = Path(__file__).parents[2] / "assets" / "videos" / f"{kernel}_animation.mp4"
    if video_path.exists():
        st.video(str(video_path))
    else:
        st.info("Manim 動畫尚未生成，請在本機執行相應腳本產生影片。")

st.caption("*注意*：在 Streamlit Community Cloud 上無法即時產生 Manim 影片，僅展示已生成的影片檔案。")
