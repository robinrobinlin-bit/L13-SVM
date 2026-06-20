import streamlit as st
import os
import numpy as np

from utils.datasets import generate_dataset
from utils.svm_model import train_svm
from utils.plotting import plot_decision_boundary

st.title("SVM 教學互動網站 - 第 3 章：互動式 SVM 實驗")

# ---- Sidebar controls ----
st.sidebar.header("模型參數")

dataset_option = st.sidebar.selectbox(
    "Dataset",
    ["blobs", "moons", "circles", "classification"],
    index=0,
    help="選擇要用來訓練 SVM 的合成資料集。",
)

kernel_option = st.sidebar.selectbox(
    "Kernel",
    ["linear", "rbf", "poly"],
    index=0,
    help="SVM 核函數，決定決策邊界的形狀。",
)

C_value = st.sidebar.slider("C (正則化參數)", 0.01, 100.0, 1.0, 0.01, format="%g")

# Gamma 只在 rbf/poly 時顯示
if kernel_option in ["rbf", "poly"]:
    gamma_value = st.sidebar.slider("Gamma", 0.001, 10.0, 1.0, 0.001, format="%g")
else:
    gamma_value = "scale"

# Degree 只在 poly 時顯示
if kernel_option == "poly":
    degree_value = st.sidebar.slider("Degree", 2, 5, 3, 1)
else:
    degree_value = 3

noise_value = st.sidebar.slider("Noise", 0.0, 0.5, 0.1, 0.01)

n_samples = st.sidebar.slider("樣本數量", 100, 500, 250, 50)

# ---- Generate data ----
@st.cache_data(show_spinner=False)
def load_data(name, n, noise, rs=42):
    X, y = generate_dataset(name, n_samples=n, noise=noise, random_state=rs)
    return X, y

X, y = load_data(dataset_option, n_samples, noise_value)

# ---- Train model ----
@st.cache_resource(show_spinner=False)
def get_model(X, y, kernel, C, gamma, degree):
    model, acc = train_svm(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)
    return model, acc

model, accuracy = get_model(X, y, kernel_option, C_value, gamma_value, degree_value)

# ---- Plot decision boundary ----
fig = plot_decision_boundary(model, X, y, title="SVM Decision Boundary")

st.plotly_chart(fig, use_container_width=True)

# ---- Display model info ----
st.subheader("模型資訊")
col1, col2 = st.columns(2)
col1.metric("訓練準確度", f"{accuracy * 100:.2f}%")
col2.metric("支援向量數量", f"{len(model.support_vectors_)} / {len(y)}")

st.markdown("---")
st.markdown("**參數說明**")
st.write(
    f"- **C**：正則化參數，值越大模型越不容忍錯誤分類，可能會 overfit。\n"
    f"- **Gamma**：核函數的寬度，值越大決策邊界會更彎曲，資料點影響範圍越小。\n"
    f"- **Degree**：多項式核的次數，次數越高模型越複雜。\n"
    f"- **Kernel**：選擇不同的核函數會改變模型能否分離非線性資料。"
)
