# 🎓 SVM Interactive Teaching Platform

**A teaching‑focused interactive web app** built with **Streamlit**, **scikit‑learn**, **Plotly**, and **Manim** to explore Support Vector Machines.

---

## 🌐 Live Demo

https://robin-l13-svm.streamlit.app

## 📦 GitHub Repository

https://github.com/robinrobinlin-bit/L13-SVM

---

## 📌 Highlights
- Interactive 2D/3D visualisations of SVM decision boundaries.
- Real‑time kernel comparison (linear, poly, rbf, sigmoid).
- Pre‑rendered Manim animations for margin and support‑vector explanations.
- Step‑by‑step dataset generation and model training.
- Debug‑friendly logging and clear UI layout.

---

## 🛠 Tech Stack
- **Streamlit** – rapid web‑app framework.
- **scikit‑learn** – SVM implementation.
- **Plotly** – interactive 2D & 3D charts.
- **Manim Community** – high‑quality educational animations.
- **Python 3.12** – core language.

---

## 📊 Workflow Diagram
![workflow](workflow.png)

---

## 📸 Screenshots
> *Add your screenshots here (e.g., `assets/screenshots/home.png`).*  
_(You can place images in `assets/screenshots/` and reference them below.)_

---

## ✨ Features
- **Dataset Generation** – blobs, moons, circles, and custom linear classification.
- **Kernel Trick Page** – switch kernels and visualise 3D decision surfaces.
- **Support Vector Visualisation** – highlighted in both 2D and 3D plots.
- **Manim‑Rendered Videos** – margin, support vectors, kernel mapping, 3D mapping.
- **Comprehensive Documentation** – prompts, log, workflow files.

---

## 📂 Project Structure
- `app.py` – main entry point.
- `src/` – data generator, SVM model wrapper, 2D plot utilities.
- `pages/` – Streamlit pages for each tutorial step.
- `manim_scenes/` – Manim scripts (rendered videos stored in `assets/videos/`).
- `requirements.txt` – Python dependencies.

---

## 📦 Installation
```bash
# Recommended: virtualenv or conda
python -m venv .venv
# Windows activation
.venv\Scripts\activate
pip install -r requirements.txt
```
> **Note:** Manim runs locally; ensure `manim-community` is installed and render videos with:
```bash
manim -pql manim_scenes/scene_01_margin.py
manim -pql manim_scenes/scene_02_support_vectors.py
manim -pql manim_scenes/scene_03_kernel_trick.py
manim -pql manim_scenes/scene_04_3d_mapping.py
```

---

## ▶️ Run the App
```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud
1. Push the repository to GitHub.
2. Go to https://share.streamlit.io and create a **New app**.
3. Connect your repo, set `app.py` as the entry file.
4. Deploy – the interactive tutorial will be live.

---

## 📄 License & Acknowledgements
- Content written in Traditional Chinese, based on scikit‑learn docs and various SVM tutorials.
- Manim animations derived from the Manim Community Edition examples.
- Contributions and issues are welcome via GitHub.
