# SVM Concept Animation + sklearn Decision Surface + Streamlit Teaching Website

## Project Overview
This repository provides a **premium, beginner‑friendly** interactive teaching website for Support Vector Machines (SVM). It combines:
- **Manim** animation videos (pre‑rendered locally) that illustrate margin, support vectors, kernel trick, and 2D→3D mapping.
- **scikit‑learn** `SVC` models for real‑time training and decision‑surface rendering.
- **NumPy** for data generation and meshgrid handling.
- **Matplotlib** for clear 2‑D decision‑boundary visualisation.
- **Plotly (graph_objects)** for an interactive 3‑D kernel‑trick concept visualisation.
- **Streamlit** for the web UI, deployable to Streamlit Community Cloud.

All dependencies are listed in `requirements.txt` **without** `manim` to keep the cloud deployment lightweight.

---

## Directory Structure
```
svm_manim_streamlit/
│   app.py
│   requirements.txt
│   README.md
│   .gitignore
│
├─ assets/
│   ├─ videos/          # pre‑rendered MP4 files from Manim
│   │   ├─ svm_margin.mp4
│   │   ├─ svm_support_vectors.mp4
│   │   ├─ svm_kernel_trick.mp4
│   │   └─ svm_3d_mapping.mp4
│   └─ images/
│       └─ svm_cover.png
│
├─ manim_scenes/
│   ├─ scene_01_margin.py
│   ├─ scene_02_support_vectors.py
│   ├─ scene_03_kernel_trick.py
│   └─ scene_04_3d_mapping.py
│
├─ src/
│   ├─ data_generator.py
│   ├─ svm_model.py
│   ├─ decision_boundary.py
│   ├─ plots_2d.py
│   ├─ plots_3d.py
│   └─ teaching_text.py
│
└─ pages/
    ├─ 0_Theory.py
    ├─ 1_Linear_SVM.py
    ├─ 2_Margin_and_Support_Vectors.py
    ├─ 3_Kernel_Trick.py
    ├─ 4_RBF_Decision_Surface.py
    └─ 5_Quiz_and_Summary.py
```

---

## Local Setup & Execution
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Run the Streamlit app locally
streamlit run app.py
```

The website will be accessible at `http://localhost:8501`.

---

## Generating Manim Animations (local only)
Manim is **not** required for deployment. To create the concept videos locally:
```bash
pip install manim

# Render each scene (high‑quality, quiet mode)
manim -pqh manim_scenes/scene_01_margin.py LinearSVMMargin
manim -pqh manim_scenes/scene_02_support_vectors.py SupportVectorsScene
manim -pqh manim_scenes/scene_03_kernel_trick.py KernelTrickScene
manim -pqh manim_scenes/scene_04_3d_mapping.py Mapping3DScene
```
Move the generated MP4 files into `assets/videos/` (the filenames above).

---

## Deploy to Streamlit Community Cloud
1. Push the repository to GitHub.
2. Ensure `requirements.txt` **does not** contain `manim`.
3. In Streamlit Cloud, create a new app, select the repository, and set **Main file path** to `app.py`.
4. Deploy – the site will load the pre‑rendered videos and all interactive components.

---

## Teaching Highlights (Chinese)
- SVM 是 **Support Vector Machine**，用於二元分類。
- **Margin**：決策邊界兩側最近點的距離，SVM 目標是最大化 margin。
- **Support Vectors**：距離邊界最近的資料點，決定模型位置。
- **C** 參數：控制錯誤容忍度，C 大 → 罰款錯誤多，模型可能過擬合。
- **Gamma** 參數：RBF kernel 的影響範圍，Gamma 大 → 邊界更彎曲、易過擬合。
- **Kernel Trick**：將資料映射到更高維空間，使非線性可分資料在新空間線性可分。此概念的 3‑D 圖僅為視覺化說明，實際的 RBF kernel 對應的是高維甚至無限維空間。

---

## License
MIT License
