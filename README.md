# -*- coding: utf-8 -*-
"""
# Online Demo: https://robin-l13-svm.streamlit.app/
# Repository: https://github.com/robinlin/L13-SVM
# Documentation:
# - [prompts.md](file:///C:/Users/user/Desktop/hw8_svm/prompts.md)
# - [log.md](file:///C:/Users/user/Desktop/hw8_svm/log.md)
# - [workflow.md](file:///C:/Users/user/Desktop/hw8_svm/workflow.md)
# - [workflow.png](file:///C:/Users/user/Desktop/hw8_svm/workflow.png)
"""README.md

# SVM 互動式教學實驗室

本專案是一套 **教學導向** 的互動式網站，使用 **Streamlit** 建立多頁應用，讓機器學習初學者能夠在瀏覽器上即時體驗 SVM（Support Vector Machine）的概念與實作。

## 目錄
- `app.py` – 主入口，設定頁面資訊與首頁內容（已內嵌於此檔）
- `src/` – 核心 Python 模組，包含資料產生、模型訓練、決策面計算與 Plotly 繪圖工具。
- `pages/` – 各章節的 Streamlit 頁面，分別說明 Linear SVM、Margin & Support Vectors、Kernel Trick、RBF Decision Surface、測驗與總結。
- `manim_scenes/` – 用 **Manim** 產生的動畫腳本（影片需自行於本機渲染，渲染後放入 `assets/videos/`，在 Cloud 上僅呈現影片檔案）。
- `requirements.txt` – 執行所需的套件。

## 安裝環境
```bash
# 建議使用 virtualenv 或 conda
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **注意**：Manim 只能在本機執行，請在本機安裝 `manim-community` 套件，並使用以下指令渲染動畫：
```bash
manim -pql manim_scenes/scene_01_margin.py
manim -pql manim_scenes/scene_02_support_vectors.py
manim -pql manim_scenes/scene_03_kernel_trick.py
manim -pql manim_scenes/scene_04_3d_mapping.py
```
渲染完成後，影片會放在 `assets/videos/`，供 Streamlit 網站播放。

## 執行網站
```bash
streamlit run app.py
```
打開瀏覽器後，左側會自動顯示所有章節的導航選單。

## 部署至 Streamlit Community Cloud
1. 把整個資料夾推送至 GitHub（確保 `requirements.txt` 已提交）。
2. 前往 https://share.streamlit.io 並登入。
3. 選擇 **New app**，連結到您的 GitHub repository 並指定 `app.py` 為入口檔。
4. 完成後即可在雲端觀看互動教學內容。

## 版權與致謝
- 本教學內容以 **繁體中文** 撰寫，參考自 scikit‑learn 官方文件與多篇 SVM 入門文章。
- Manim 動畫參考自 Manim Community Edition 開源範例。
- 若有任何建議或 bug 回報，歡迎提交 Issue 至 GitHub。
"""
