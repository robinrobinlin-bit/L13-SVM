# Prompts for SVM Interactive Teaching Project

## Initial SVM Teaching Website Prompt
- Build an interactive Streamlit website that teaches Support Vector Machines (SVM) concepts.
- Include pages for linear SVM, margin & support vectors, kernel trick, RBF decision surface, and a quiz.
- Use scikit‑learn for model training and Plotly for decision‑boundary visualisation.
- Provide pre‑rendered Manim animations for each concept.

## Key Modifications (10‑15 prompts) Used During Development
1. **Home routing conflict** – Resolve duplicate route names in Streamlit navigation.
2. **Result undefined** – Fix undefined variable `result` in the interactive page.
3. **Dataset classification addition** – Add `"classification": generate_linear_data` to dataset dispatcher.
4. **Manim API update** – Replace deprecated `axes.get_graph` with `axes.plot` for Manim 0.20.1.
5. **3D coordinate fix** – Change 2‑D coordinate arrays to 3‑D `[x, y, 0]` in `KernelTrickScene`.
6. **Video conditional rendering** – Show intro videos only if files exist, otherwise display info.
7. **GitHub deployment instructions** – Document steps for deploying to Streamlit Community Cloud.
8. **Streamlit page imports** – Ensure `streamlit as st` is imported in new pages.
9. **Plotly interactivity** – Connect scikit‑learn model updates to Plotly charts in real time.
10. **Manim pre‑render workflow** – Automate rendering of Manim scenes and placement in `assets/videos/`.
11. **Debugging tip** – Use `python -m compileall` to verify syntax after each change.
12. **README enhancement** – Add demo link, repository link, and documentation references.
13. **Workflow diagram creation** – Produce Mermaid diagrams for the project pipelines.
14. **Matplotlib info graphic** – Generate a blue‑red colour key image describing components.
15. **Prompt documentation** – Compile all prompts into `prompts.md` for future reference.

## Feature Overview
- **Five‑page integration** covering theory, margin, kernel trick, RBF, and quiz.
- **Manim pre‑rendered videos** for margin, support vectors, and kernel trick.
- **Debugging and GitHub workflow** documented for reproducibility.
- **Deployment guide** for Streamlit Cloud.
