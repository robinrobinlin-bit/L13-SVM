import os

# Create project directories (will be created automatically when files are written)
for folder in [
    "assets/videos",
    "manim_scenes",
    "utils",
    "pages",
]:
    os.makedirs(folder, exist_ok=True)
