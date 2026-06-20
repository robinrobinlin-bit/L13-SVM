# pages/Steps.py
"""
SVM 教學步驟說明頁面 – 以條列方式呈現 SVM 基本概念流程。
此頁可由側邊欄或首頁連結導向，讓使用者快速了解 SVM 的核心步驟。
"""

import streamlit as st

st.title("📚 SVM 教學步驟概覽")

steps = [
    "**Step 1：分類問題** – 先有一組帶標籤的二元資料 (X, y)。",
    "**Step 2：找一條分隔線** – 目標是找到一條可以將兩類資料分開的直線（或超平面）。",
    "**Step 3：不是隨便一條線，而是 margin 最大的線** – SVM 會選取使兩類最近點之間的間隔（margin）最大的分隔線。",
    "**Step 4：靠近邊界的點叫 support vectors** – 位於 margin 兩側、最貼近分隔線的資料點稱為 support vectors，它們決定了最終模型。",
    "**Step 5：資料線性不可分時，使用 kernel trick** – 透過核函式將資料映射至更高維空間，使原本不可分的資料在新空間中變得線性可分。",
    "**Step 6：RBF kernel 可以產生非線性決策邊界** – 常用的徑向基 (RBF) 核會在高維空間中產生彎曲的決策面，對應回 2D 時形成非線性邊界。",
    "**Step 7：sklearn SVC 實作** – 最後使用 scikit‑learn 的 `SVC` 進行模型訓練、預測與可視化。",
]

for i, txt in enumerate(steps, start=1):
    st.markdown(f"{i}. {txt}")

st.info("在左側導覽列選擇其他章節以深入了解每一步驟的實作與互動示範。")
