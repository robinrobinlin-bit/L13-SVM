import streamlit as st
import os

st.title("SVM 教學互動網站 - 第 2 章：Margin 與 Support Vectors")
st.markdown("""
## Margin 與 Support Vectors

- **決策函數**: $f(\mathbf{x}) = \mathbf{w}\cdot\mathbf{x} + b$
- **分類規則**: $\text{sign}(f(\mathbf{x}))$
- **Margin**: $\frac{2}{\|\mathbf{w}\|}$
- **最佳化目標**:
  \[\min_{\mathbf{w},b}\ \frac{1}{2}\|\mathbf{w}\|^2\]
  subject to $y_i(\mathbf{w}\cdot\mathbf{x}_i + b) \ge 1$

這段公式說明了 SVM 如何透過 **最大化 margin** 來得到最好的分隔平面。
""")

video_path = os.path.join("assets", "videos", "support_vectors_intro.mp4")
if os.path.exists(video_path):
    st.video(video_path)
else:
    st.info("請先使用 Manim 產生 `support_vectors_intro.mp4` 並放置於 `assets/videos/` 資料夾。")
