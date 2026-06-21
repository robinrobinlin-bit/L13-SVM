# -*- coding: utf-8 -*-
"""pages/3_Kernel_Trick.py
說明 SVM 核技巧（Kernel Trick），提供互動式切換不同 kernel 的視覺化。
"""

import streamlit as st
import numpy as np
from pathlib import Path

from src.data_generator import make_dataset
from src.svm_model import get_model
from src.plots_2d import plot_decision_boundary_2d
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px


def run():
    st.title("3️⃣ Kernel Trick")
    st.markdown(
        """
        **Kernel Trick** 允許 SVM 在非線性可分的資料上使用核函式，將特徵映射到更高維度，使資料在新空間中線性可分。
        常見的 kernel 包括 `rbf`（徑向基函式）與 `poly`（多項式核）。
        """
    )

    # 側欄參數設定
    st.sidebar.header("🛠 參數設定")
    dataset_name = st.sidebar.selectbox("資料集", ["blobs", "moons", "circles", "classification"], index=0)
    n_samples = st.sidebar.slider("樣本數量", 100, 500, 250, step=50)
    noise = st.sidebar.slider("噪聲", 0.0, 0.5, 0.1, step=0.05)
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"], index=1)
    C = st.sidebar.slider("C (正則化)", 0.01, 10.0, 1.0, step=0.01)
    # gamma 只在 rbf、poly 時顯示
    if kernel in ["rbf", "poly"]:
        gamma = st.sidebar.slider("Gamma", 0.001, 5.0, 1.0, step=0.001)
    else:
        gamma = "scale"
    # degree 只在 poly 時顯示
    if kernel == "poly":
        degree = st.sidebar.slider("Degree", 2, 5, 3, step=1, key="degree")
    else:
        degree = 3
    # Kernel for 3D Surface selection
    kernel_3d = st.sidebar.selectbox(
        "Kernel for 3D Surface", ["linear", "poly", "rbf", "sigmoid"],
        index=0, key="kernel_3d"
    )
    C_3d = st.sidebar.slider("C (3D)", 0.01, 10.0, 1.0, step=0.01, key="C_3d")
    if kernel_3d in ["rbf", "sigmoid"]:
        gamma_3d = st.sidebar.slider("Gamma (3D)", 0.001, 5.0, 1.0, step=0.001, key="gamma_3d")
    else:
        gamma_3d = "scale"
    if kernel_3d == "poly":
        degree_3d = st.sidebar.slider("Degree (3D)", 2, 5, 3, step=1, key="degree_3d")
    else:
        degree_3d = 3

    # 產生資料
    X, y, _ = make_dataset(dataset_name, n_samples=n_samples, noise=noise, random_state=42)

    # 訓練模型
    model, accuracy = get_model(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)
    # 3D model using selected kernel for surface and support vectors
    model_3d, _ = get_model(X, y, kernel=kernel_3d, C=C_3d, gamma=gamma_3d, degree=degree_3d)

    # 繪圖（使用 2D 版）
    fig = plot_decision_boundary_2d(model, X, y, title=f"Kernel = {kernel.upper()}")
    st.plotly_chart(fig, use_container_width=True)

    # 模型資訊
    st.subheader("模型資訊")
    st.write(f"**Kernel**: {kernel}")
    st.write(f"**訓練準確度**: {accuracy * 100:.2f}%")
    st.write(f"**支援向量數量**: {len(model.support_vectors_)} / {len(y)}")
    # 3D Decision Function Surface
    try:
        # Create meshgrid using original data
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = model_3d.decision_function(grid).reshape(xx.shape)

        fig3d = go.Figure()
        fig3d.add_surface(x=xx, y=yy, z=zz, colorscale="RdBu", showscale=False, opacity=0.7)
        # Overlay data points
        fig3d.add_scatter3d(
            x=X[:, 0],
            y=X[:, 1],
            z=np.zeros_like(y),
            mode="markers",
            marker=dict(
                color=["red" if label == 0 else "blue" for label in y],
                size=4,
            ),
            name="Data",
        )
        # Overlay support vectors (if any)
        try:
            sv = model_3d.support_vectors_
            fig3d.add_scatter3d(
                x=sv[:, 0],
                y=sv[:, 1],
                z=np.zeros(sv.shape[0]),
                mode="markers",
                marker=dict(
                    color="yellow",
                    size=8,
                    symbol="diamond",
                ),
                name="Support Vectors",
            )
        except Exception:
            # No support vectors available for the selected kernel
            st.info("Support vectors not available for this kernel.")
        fig3d.update_layout(
            scene=dict(
                xaxis_title="X₁",
                yaxis_title="X₂",
                zaxis_title="Decision Function",
            ),
            title="3D Decision Function Surface",
            margin=dict(l=0, r=0, b=0, t=30),
            annotations=[
                dict(
                    text="Z 軸代表 SVM decision_function 分數，Z=0 附近就是決策邊界。",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=0,
                    showarrow=False,
                    font=dict(size=12),
                ),
                dict(
                    text="不同 kernel 會改變 decision_function 的形狀",
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=0.1,
                    showarrow=False,
                    font=dict(size=12),
                ),
            ],
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # RBF Kernel Feature Mapping
        if kernel_3d == "rbf":
            st.subheader("RBF Kernel Feature Mapping")
            st.markdown("""RBF kernel 透過將資料映射到高維空間，使原本無法線性分割的資料能被超平面分開。""")
            if dataset_name in ["circles", "moons"]:
                st.info("非線性資料集（circles/moons）展示 RBF 曲面效果。")
            # Ensure meshgrid exists (reuse xx, yy if already defined)
            try:
                _ = xx
                _ = yy
            except NameError:
                x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100)
                )
                grid = np.c_[xx.ravel(), yy.ravel()]
                zz = model_3d.decision_function(grid).reshape(xx.shape)
            else:
                zz = model_3d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            fig_rbf = go.Figure()
            fig_rbf.add_surface(x=xx, y=yy, z=zz, colorscale="Viridis", showscale=False, opacity=0.8)
            # Overlay data points
            fig_rbf.add_scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=np.zeros_like(y),
                mode="markers",
                marker=dict(
                    color=["red" if label == 0 else "blue" for label in y],
                    size=4,
                ),
                name="Data",
            )
            # Overlay support vectors if available
            try:
                sv = model_3d.support_vectors_
                fig_rbf.add_scatter3d(
                    x=sv[:, 0],
                    y=sv[:, 1],
                    z=np.zeros(sv.shape[0]),
                    mode="markers",
                    marker=dict(color="yellow", size=8, symbol="diamond"),
                    name="Support Vectors",
                )
            except Exception:
                st.info("Support vectors not available for RBF surface.")
            fig_rbf.update_layout(
                scene=dict(xaxis_title="X₁", yaxis_title="X₂", zaxis_title="Decision Function"),
                title="RBF Kernel Decision Surface",
                margin=dict(l=0, r=0, b=0, t=30),
            )
            st.plotly_chart(fig_rbf, use_container_width=True)

        # Kernel Comparison Dashboard
        st.subheader("Kernel Comparison Dashboard")
        st.markdown("""
        不同 kernel 會改變資料在特徵空間中的分離方式，因此 accuracy 與 support vector 數量會不同。
        """)

        # Train models for each kernel and evaluate
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        results = []
        for k in kernels:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
                # Set kernel‑specific parameters
                if k == "poly":
                    model_k, _ = get_model(
                        X_train,
                        y_train,
                        kernel=k,
                        C=C,
                        degree=degree,
                        gamma=gamma if gamma != "scale" else "scale",
                    )
                else:
                    model_k, _ = get_model(
                        X_train,
                        y_train,
                        kernel=k,
                        C=C,
                        gamma=gamma if k in ["rbf", "sigmoid"] else "scale",
                    )
                preds = model_k.predict(X_test)
                acc = accuracy_score(y_test, preds)
                sv_cnt = len(model_k.support_vectors_)
                results.append({"Kernel": k, "Test Accuracy": acc, "Support Vectors": sv_cnt})
            except Exception as e:
                st.info(f"Kernel {k} comparison failed: {e}")
                results.append({"Kernel": k, "Test Accuracy": None, "Support Vectors": None})

        # Show results table
        if results:
            df = pd.DataFrame(results)
            st.dataframe(df.style.format({"Test Accuracy": "{:.2%}"}))
            # Bar chart of accuracies
            fig_bar = px.bar(
                df,
                x="Kernel",
                y="Test Accuracy",
                text="Test Accuracy",
                title="Kernel Test Accuracy Comparison",
            )
            fig_bar.update_traces(texttemplate="%{text:.2%}", textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)
    except Exception as e:
        st.info(f"3D decision surface could not be displayed: {e}")
    # Kernel Trick intro video
    video_path = Path(__file__).parent.parent / "assets" / "videos" / "kernel_trick_intro.mp4"
    if video_path.is_file():
        st.video(str(video_path))
    else:
        st.info(f"Video not found: {video_path.name}")

run()
