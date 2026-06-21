# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(
    page_title="SVM 互動式教學網站",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 首頁內容（已從 pages/0_Home.py 移至此）
st.title("Home｜首頁")
st.markdown("""
## 專案介紹
本教學平台以互動式方式說明 Support Vector Machines (SVM) 的概念與實作，適合機器學習初學者。

### 學習路徑
1. 線性 SVM（Linear SVM）
2. 邊界與支援向量（Margin & Support Vectors）
3. 核技巧（Kernel Trick）
4. RBF 決策面探索（RBF Decision Surface）
5. 測驗與總結（Quiz & Summary）
""")

# 側欄快速導航按鈕（使用 session_state 切換頁面）
for i, (title, page) in enumerate([
    ("線性 SVM", "Linear_SVM"),
    ("邊界與支援向量", "Margin_and_Support_Vectors"),
    ("核技巧", "Kernel_Trick"),
    ("RBF 決策面", "RBF_Decision_Surface"),
    ("測驗與總結", "Quiz_and_Summary"),
]):
    if st.button(title, key=f"home_nav_{i}"):
        st.session_state["page"] = page

st.info("使用左側側欄也可切換頁面。")
