import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.svm import SVC
import plotly.graph_objects as go

def main():
    st.set_page_config(page_title="SVM 教學互動網站", layout="wide")
    st.title("SVM 教學互動網站 (MVP)")
    
    st.sidebar.title("章節選單")
    page = st.sidebar.radio("選擇頁面", ["首頁", "SVM 互動演示"])

    if page == "首頁":
        st.write("這是首頁，使用左側選單切換不同章節。")
        st.write("本 MVP 實作了 SVM 的互動視覺化，幫助理解超平面與邊界。")
    
    elif page == "SVM 互動演示":
        st.header("SVM 互動演示")
        
        # Sidebar controls
        C = st.sidebar.slider("C (正規化強度)", 0.01, 10.0, 1.0)
        kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf"])
        
        # Generate dummy data
        X, y = datasets.make_blobs(n_samples=50, centers=2, random_state=6, cluster_std=0.60)
        
        # Fit SVM
        clf = SVC(kernel=kernel, C=C)
        clf.fit(X, y)
        
        # Visualization
        fig = go.Figure()
        
        # Add points
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', 
                                 marker=dict(color=y, colorscale='Viridis', size=10)))
        
        # Create grid for decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Add contour
        fig.add_trace(go.Contour(x=np.linspace(x_min, x_max, 50), y=np.linspace(y_min, y_max, 50), z=Z, 
                                 colorscale='RdBu', opacity=0.3, showscale=False, contours=dict(start=-1, end=1, size=0.1)))
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
