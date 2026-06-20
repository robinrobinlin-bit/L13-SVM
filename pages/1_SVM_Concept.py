# pages/1_SVM_Concept.py
"""
SVM 概念說明頁面（第 1 章）
"""
import streamlit as st
import os

def run():
    st.title("SVM 教學互動網站 - 第 1 章：SVM 是什麼？")
    st.markdown(
        """
        ## 什麼是 Support Vector Machine (SVM)
        
        - 二元分類器，尋找能把兩類資料分開的 **超平面**。
        - 目標是 **最大化 margin**（兩類最近點之間的距離），
          這樣的模型在新資料上具有較好的泛化能力。
        - 支援向量 (Support Vectors) 是離超平面最近的點，
          它們決定了最終的決策邊界。
        """
    )
    video_path = os.path.join("assets", "videos", "svm_margin_intro.mp4")
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.info("請先使用 Manim 產生 `svm_margin_intro.mp4` 並放置於 `assets/videos/` 資料夾。")
