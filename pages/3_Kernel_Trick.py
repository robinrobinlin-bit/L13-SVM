# pages/3_Kernel_Trick.py
"""
Kernel Trick 教學頁面 – 第 3 頁

顯示 Manim 產生的 Kernel Trick 概念動畫，並提供簡易的參數控制讓使用者可以在本機重新產生影片。
"""

import streamlit as st
from pathlib import Path
from src import manim_runner

st.title("3️⃣ Kernel Trick (核技巧)")

st.write(
    "在此章節，我們說明為什麼線性模型在某些資料上無法分割，\n"
    "以及透過 **Kernel Trick** 將資料映射到更高維度後，\n"
    "在 3D 空間中可以用平面分割，最終得到非線性決策邊界。"
)

# ---------- 影片展示 ----------
video_path = Path(__file__).parents[2] / "assets" / "videos" / "svm_kernel_trick.mp4"
if video_path.exists():
    st.video(str(video_path))
else:
    st.warning("Kernel Trick 動畫尚未生成，請先在本機執行 Manim 產生影片。")

# ---------- 參數控制 (本機重新產生) ----------
st.subheader("重新產生影片（本機）")
st.caption("此功能僅在本機執行，部署至 Streamlit Cloud 時會顯示無法產生的訊息。")

show_axes = st.checkbox("顯示座標軸", value=True)
font_scale = st.slider("字體縮放", 0.5, 2.0, 1.0, 0.1)

if st.button("產生影片"):
    success, msg = manim_runner.generate_video(
        scene_file="manim_scenes/scene_03_kernel_trick.py",
        class_name="KernelTrickScene",
    )
    if success:
        st.success("影片已產生，重新載入中…")
        if video_path.exists():
            st.video(str(video_path))
    else:
        st.error(f"產生影片失敗: {msg}")

# ---------- 文字說明 ----------
st.subheader("Kernel Trick 核心概念")
st.markdown(
    "- **非線性不可分**：在 2D 平面中，某些資料（如同心圓）無法用直線分割。\n"
    "- **映射函式**：以 `z = x^2 + y^2` 為例，將 2D 點映射到 3D 空間。\n"
    "- **線性可分**：在 3D 空間中，資料可以被一個平面分開。\n"
    "- **回投至 2D**：投影回原始平面得到非線性的決策邊界。"
)
st.warning("**注意**：此 3D 圖僅為 Kernel Trick 的概念視覺化，並不代表 RBF SVM 真正映射至 3 維。實際上 RBF kernel 對應的是非常高維甚至無限維的特徵空間。")
