# src/manim_runner.py
"""
Utility module for invoking Manim from Streamlit (local only).

The workflow:
1. User 在 Streamlit 側調整參數或選擇要產生的動畫。
2. 按下 **Generate Video** 按鈕時，呼叫 `generate_video()`。
3. `generate_video()` 會在本機以 `subprocess.run(['manim', ...])` 執行 Manim，產生 mp4 檔案於 `assets/videos/`。
4. 完成後返回 True，讓 Streamlit 重新載入影片。

**重要**：此功能僅在本機執行，若部署至 Streamlit Cloud，會顯示提示說明「Manim 只能本機產生，雲端無法執行」。
"""

import subprocess
from pathlib import Path
from typing import Tuple

# 影片輸出資料夾（相對於專案根目錄）
ASSETS_VIDEOS = Path(__file__).parents[2] / "assets" / "videos"


def generate_video(scene_file: str, class_name: str) -> Tuple[bool, str]:
    """在本機執行 Manim，生成 MP4 檔案。

    參數說明:
    - scene_file: 相對於專案根目錄的 Python 檔案路徑，例如 `manim_scenes/scene_01_margin.py`。
    - class_name: Manim scene class 名稱，例如 `LinearSVMMargin`。

    回傳:
    - success: 是否成功產生影片。
    - video_path: 產生的影片完整路徑（若成功），或錯誤訊息。
    """
    # 構造絕對路徑以避免 cwd 問題
    project_root = Path(__file__).parents[2]
    scene_path = project_root / scene_file
    if not scene_path.exists():
        return False, f"Scene file not found: {scene_path}"

    # 執行 Manim 指令，-pqh = preview, high quality, quiet output
    cmd = [
        "manim",
        "-pqh",
        str(scene_path),
        class_name,
    ]
    try:
        # 使用 subprocess.run，捕獲 stdout 以免佔用過多終端畫面
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        return False, f"Exception while running manim: {e}"

    if result.returncode != 0:
        return False, f"Manim failed (code {result.returncode}): {result.stderr}"

    # Manim 會在 `media/videos` 下產生 mp4，我們把它搬到 assets/videos
    # 預設路徑格式: media/videos/<scene_name>/<class_name>.mp4
    generated_mp4 = project_root / "media" / "videos" / Path(scene_file).stem / f"{class_name}.mp4"
    if not generated_mp4.exists():
        return False, f"Generated video not found: {generated_mp4}"

    # 確保 assets/videos 資料夾存在
    ASSETS_VIDEOS.mkdir(parents=True, exist_ok=True)
    dest_path = ASSETS_VIDEOS / f"{class_name}.mp4"
    # 複製或覆寫既有檔案
    dest_path.write_bytes(generated_mp4.read_bytes())
    return True, str(dest_path)
