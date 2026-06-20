# app.py
"""Streamlit 主入口，負責載入各個教學頁面並提供全域樣式。
此檔案使用動態 import 機制，根據側邊欄的選單載入 pages 目錄下的 Python 模組。
所有中文說明、註解均採用繁體中文，並套用 Inter 字體與深色主題以提升視覺品質。
"""
import importlib
import os
import streamlit as st

# ---------- 全域樣式 ----------
# 使用 Google Fonts 的 Inter，並設定深色主題
st.set_page_config(page_title="SVM 教學互動網站", layout="centered")

# 自訂 CSS（包含字體與配色）
custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        html, body, div, span, applet, object, iframe,
        h1, h2, h3, h4, h5, h6, p, blockquote, pre,
        a, abbr, acronym, address, big, cite, code,
        del, dfn, em, img, ins, kbd, q, s, samp,
        small, strike, strong, sub, sup, tt, var,
        b, u, i, center,
        dl, dt, dd, ol, ul, li,
        fieldset, form, label, legend,
        table, caption, tbody, tfoot, thead, tr, th, td {
            font-family: 'Inter', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #1e1e2f, #2e2e3f);
            color: #f0f0f0;
        }
        .stSidebar {
            background: #252539;
        }
        .stButton>button {
            background-color: #4e79a7;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stSlider>div > div > div > div {
            background: #4e79a7;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ---------- 側邊欄選單 ----------
st.sidebar.title("SVM 教學導航")
page_options = {
    "0. 理論概述": "pages.0_Theory",
    "1. 線性 SVM": "pages.1_Linear_SVM",
    "2. Margin 與 Support Vectors": "pages.2_Margin_and_Support_Vectors",
    "3. Kernel Trick": "pages.3_Kernel_Trick",
    "4. RBF Decision Surface": "pages.4_RBF_Decision_Surface",
    "5. 小測驗 & 總結": "pages.5_Quiz_and_Summary",
}
selected = st.sidebar.radio("選擇頁面", list(page_options.keys()))
module_path = page_options[selected]

# 動態載入並執行
module = importlib.import_module(module_path)
if hasattr(module, "run"):
    module.run()
else:
    st.error("該頁面缺少 run() 函式")
