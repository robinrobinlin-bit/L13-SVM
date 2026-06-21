# Development Log

## Chronological Development Process

### 1. Home Routing Conflict
- **Problem**: Duplicate navigation entries for Home page caused routing conflict.
- **Cause**: Two pages defined the same title and default entry.
- **Fix**: Renamed Home page file and updated navigation keys.
- **Verification**: Only a single Home entry appears in the app.

### 2. `result` Undefined Error
- **Problem**: `NameError: name 'result' is not defined` on interactive page.
- **Cause**: Variable referenced before assignment.
- **Fix**: Initialized `result = None` before use.
- **Verification**: Page runs without exception.

### 3. Dataset Classification Feature
- **Problem**: No "classification" option in dataset dispatcher.
- **Fix**: Added `"classification": generate_linear_data` to `dispatch` in `utils/datasets.py`.
- **Verification**: `generate_dataset("classification")` works.

### 4. Manim API Update (0.20.1)
- **Problem**: `axes.get_graph` deprecated, raising `TypeError`.
- **Fix**: Replaced all calls with `axes.plot` preserving arguments.
- **Verification**: `python -m compileall manim_scenes` succeeds; scenes render.

### 5. 3D Coordinate Errors in Kernel Trick
- **Problem**: Broadcast error when using 2‑D coordinates in 3‑D scene.
- **Fix**: Updated all coordinate arrays to `[x, y, 0]`.
- **Verification**: Compilation succeeds, animation displays correctly.

### 6. Conditional Video Rendering
- **Problem**: Streamlit pages attempted to play missing video files.
- **Fix**: Added `pathlib.Path` checks; use `st.video()` if file exists, else `st.info()`.
- **Verification**: Pages load gracefully with or without video files.

### 7. GitHub → Streamlit Cloud Deployment Documentation
- **Problem**: No clear deployment guide.
- **Fix**: Added steps in the README and documented workflow in `workflow.md`.
- **Verification**: New contributors can deploy successfully.

All changes have been compiled and verified with `python -m compileall` across relevant directories.
