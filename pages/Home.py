# pages/Home.py
"""
首頁 – 什麼是 SVM？

此頁提供 SVM 的概念介紹、主要術語說明與整體教學流程概覽。
使用簡潔的中文說明與圖示，讓使用者快速了解本教學網站的內容。
"""

import streamlit as st
from pathlib import Path

st.title("🤖 SVM 教學互動網站")

st.header("什麼是 Support Vector Machine (SVM)？")
st.write(
    "Support Vector Machine（支援向量機）是一種監督式機器學習模型，\n"
    "主要用於二元分類（也可擴展至多類別與回歸）。\n"
    "SVM 透過在特徵空間中尋找最大化 **margin**（間隔）的決策超平面，\n"
    "使得分類效果在訓練資料上具備良好的魯棒性。"
)

st.subheader("核心概念")
st.markdown(
    "- **決策邊界 (Decision Boundary)**：將不同類別資料分開的超平面。\n"
    "- **Margin**：決策邊界與最近資料點（Support Vectors）之間的距離。\n"
    "- **Support Vectors**：位於 margin 兩側、最貼近決策邊界的資料點，\n"
    "  它們決定了最終的分類平面。\n"
    "- **Kernel Trick**：透過核函式將資料映射至更高維空間，使非線性可分問題在該空間變為線性可分。"
)

st.subheader("教學流程概覽")
st.markdown(
    "1️⃣ **Linear SVM** – 觀看線性決策線、Margin、Support Vectors。\n"
    "2️⃣ **Margin 與 Support Vectors** – 深入說明 Margin 的意義與支援向量的角色。\n"
    "3️⃣ **Kernel Trick** – 了解將 2D 資料映射至 3D 的概念示意。\n"
    "4️⃣ **RBF Kernel 互動展示** – 調整 C、gamma 觀察非線性決策面。\n"
    "5️⃣ **sklearn 真實決策面** – 使用 sklearn SVC 產生實際決策邊界與支援向量。\n"
    "6️⃣ **小測驗 / 學習總結** – 檢測學習成效，提供重點整理。"
)

st.info(
    "點選左側導覽列即可進入各章節，並透過滑桿即時調整模型參數，觀察 SVM 的行為變化。"
)

# 顯示封面圖片（如果存在）
cover_path = Path(__file__).parents[2] / "assets" / "images" / "svm_cover.png"
if cover_path.exists():
    st.image(str(cover_path), caption="SVM 教學概覽", use_column_width=True)
else:
    st.warning("封面圖片尚未上傳，請將 svm_cover.png 放到 assets/images/ 目錄。")
