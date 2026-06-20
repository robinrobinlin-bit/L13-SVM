import streamlit as st
import os
import numpy as np

from utils.datasets import generate_dataset
from utils.svm_model import train_svm
from utils.plotting import plot_decision_boundary
from src import manim_runner

st.title("SVM 教學互動網站 - 第 4 章：Kernel Trick")

st.markdown("""
## Kernel Trick 直覺說明

- **Linear kernel** 只能分離線性可分的資料。
- **RBF kernel** 會將資料映射到更高維度，使原本不可分的資料在新空間變得線性可分。
- **Polynomial kernel** 用多項式特徵提升維度，`degree` 越高模型越複雜。
""")

# 影片播放（如果有）
video_path = os.path.join("assets", "videos", "svm_kernel_trick.mp4")
if os.path.exists(video_path):
    st.video(str(video_path))
else:
    st.info("請先使用 Manim 產生 `svm_kernel_trick.mp4` 並放置於 `assets/videos/` 資料夾。")

st.sidebar.header("互動參數")

dataset_option = st.sidebar.selectbox(
    "Dataset (非線性示例)",
    ["moons", "circles"],
    index=0,
)

kernel_option = st.sidebar.selectbox(
    "Kernel",
    ["linear", "rbf", "poly"],
    index=1,
)

C_value = st.sidebar.slider("C", 0.01, 100.0, 1.0, 0.01)

if kernel_option in ["rbf", "poly"]:
    gamma_value = st.sidebar.slider("Gamma", 0.001, 10.0, 1.0, 0.001)
else:
    gamma_value = "scale"

# ------------------- 動畫參數 (示例) -------------------
anim_param = st.sidebar.slider("Animation parameter", 0.0, 5.0, 1.0, 0.1)
if "prev_anim_param" not in st.session_state:
    st.session_state.prev_anim_param = anim_param

if anim_param != st.session_state.prev_anim_param:
    with st.spinner("產生 Manim 動畫..."):
        success, msg = manim_runner.generate_video(
            scene_file="manim_scenes/scene_03_kernel_trick.py",
            class_name="KernelTrickScene",
        )
        if success:
            st.success("影片已產生，重新載入中…")
        else:
            st.error(f"產生影片失敗: {msg}")
    st.session_state.prev_anim_param = anim_param

if kernel_option == "poly":
    degree_value = st.sidebar.slider("Degree", 2, 5, 3, 1)
else:
    degree_value = 3

noise_value = st.sidebar.slider("Noise", 0.0, 0.5, 0.1, 0.01)

n_samples = st.sidebar.slider("樣本數量", 100, 500, 250, 50)

@st.cache_data(show_spinner=False)
def load_data(name, n, noise, rs=42):
    X, y = generate_dataset(name, n_samples=n, noise=noise, random_state=rs)
    return X, y

X, y = load_data(dataset_option, n_samples, noise_value)

@st.cache_resource(show_spinner=False)
def get_model(X, y, kernel, C, gamma, degree):
    model, acc = train_svm(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)
    return model, acc

model, accuracy = get_model(X, y, kernel_option, C_value, gamma_value, degree_value)

fig = plot_decision_boundary(model, X, y, title=f"Kernel: {kernel_option}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("模型資訊")
st.write(f"訓練準確度: {accuracy*100:.2f}%")
st.write(f"支援向量數量: {len(model.support_vectors_)} / {len(y)}")

st.markdown("---")
st.markdown("**參數說明**")
st.write(
    f"- **Kernel**：決定資料映射方式。\n"
    f"- **C**：正則化參數，值大會降低錯誤容忍度，可能 overfit。\n"
    f"- **Gamma**：RBF/Poly 核的寬度，值大會使邊界更彎曲。\n"
    f"- **Degree**：Poly 核的次方，次數越高模型越複雜。"
)
