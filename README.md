# SVM Teaching Interactive Streamlit App

## 目標
建立一個 **可部署至 Streamlit Community Cloud** 的教學網站，用於說明 Support Vector Machine（SVM）的概念、數學直覺、margin、support vectors、kernel trick，並提供使用者即時調整參數、觀察決策邊界的互動體驗。

## 目錄結構
```
svm-streamlit-teaching/   (本專案根目錄)
│
├─ app.py                     # Streamlit 入口，會自動載入 pages/*.py
├─ requirements.txt           # 部署所需的 Python 套件（不含 manim）
├─ .gitignore                 # 常見的 Python / Streamlit 忽略檔案
├─ README.md                  # 本檔案
│
├─ assets/
│   └─ videos/                # 渲染好的 mp4 / webm（若缺失會顯示提示）
│
├─ manim_scenes/              # 本機開發用的 Manim 腳本 (不會部署)
│   ├─ svm_margin_scene.py
│   ├─ support_vectors_scene.py
│   └─ kernel_trick_scene.py
│
├─ utils/                     # 模組化程式碼
│   ├─ __init__.py            # 讓 utils 成為套件
│   ├─ datasets.py
│   ├─ svm_model.py
│   ├─ plotting.py
│   └─ explanations.py        # 目前為空，可自行加入中文說明文字
│
└─ pages/                     # Streamlit 多頁面
    ├─ 1_SVM_Concept.py
    ├─ 2_Margin_and_Support_Vectors.py
    ├─ 3_Interactive_SVM.py
    ├─ 4_Kernel_Trick.py
    └─ 5_Quiz.py
```

## 如何在本機執行
```bash
# 1. 進入專案目錄
cd C:\Users\user\Desktop\hw8_svm

# 2. 建立虛擬環境（建議）
python -m venv .venv
.\.venv\Scripts\activate   # PowerShell / CMD

# 3. 安裝套件
pip install -r requirements.txt

# 4. 執行 Streamlit
streamlit run app.py
```
瀏覽器會自動開啟 `http://localhost:8501`，左側選單即為 1~5 頁。

## Manim 影片製作（本機）
1. 安裝 Manim（若未安裝）
```bash
pip install manim
```
2. 在 `manim_scenes/` 編寫動畫腳本（已提供空白範本）
3. 渲染指令範例：
```bash
manim -pqh manim_scenes/svm_margin_scene.py SVMMarginScene
manim -pqh manim_scenes/support_vectors_scene.py SupportVectorsScene
manim -pqh manim_scenes/kernel_trick_scene.py KernelTrickScene
```
4. 產生的影片放入 `assets/videos/`，必須命名為：
   - `svm_margin_intro.mp4`
   - `support_vectors_intro.mp4`
   - `kernel_trick_intro.mp4`
   若未放入，對應頁面會顯示提醒文字。

## 部署至 Streamlit Community Cloud
1. **Push 專案到 GitHub**（或其他 Git 平台）
```bash
git init
git add .
git commit -m "Initial MVP"
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```
2. 前往 https://share.streamlit.io，登入後點 **New app** → 連結您的 repo → 設定 **Main file path** 為 `app.py`。
3. 若需要固定 Python 版本，於 Streamlit Cloud 設定頁面選擇 **Python 3.11**（建議）。
4. 點 **Deploy**，系統會自動安裝 `requirements.txt`，完成後即可取得公開 URL。
5. **影片注意**：若 Cloud 上找不到 `assets/videos/*.mp4`，會顯示提示。您可以將大檔影片上傳至 YouTube / Google Drive / Cloudflare，然後把 `st.video()` 裡的路徑改成對應的 URL（只要在 `pages/` 中手動修改即可）。

## 常見問題排除 (FAQ)
| 問題 | 解決方法 |
|------|----------|
| **影片不顯示** | 確認 `assets/videos/` 中有正確的檔名，或在 `pages/*.py` 中把 `st.video()` 改成 `st.video(<外部URL>)` |
| **部署失敗 – 缺少系統套件** | 本 MVP 不需要任何系統依賴；若您在 Cloud 上自行加入 `manim`，請參考官方 `packages.txt`（FFmpeg、Cairo、Pango 等） |
| **Plotly 圖表卡住** | 資料量上限已限制在 500，使用 `st.cache_data` / `st.cache_resource`，若仍慢可減少 `n_samples` 或改用 `resolution=0.05` 修改 `make_meshgrid` 的 `h` 參數 |
| **程式報錯 `module not found: utils`** | 確認執行 `streamlit run app.py` 時的工作目錄是專案根目錄，或在 `app.py` 前加 `import sys, os; sys.path.append(os.path.abspath('.'))`（已在 `app.py` 中隱式完成） |

## 後續可加入的加分功能
- 參數比較表與 kernel 適用情境表格（Markdown 放在適當頁面）
- 圖表下載按鈕：`st.download_button` 下載 Plotly 圖為 PNG
- 支援向量比例（`len(model.support_vectors_) / len(y)`）顯示於模型資訊區塊
- 小測驗分數統計與回饋文字（目前已顯示總分）
- 每頁底部「下一步建議」文字區塊
- 3D Plotly 互動（在 `4_Kernel_Trick.py` 中加入 `go.Surface`）

---
**祝您開發順利！如需進一步功能（如第 4/5 頁的 UI 美化、3D 可視化等），隨時告訴我，我會立即補上程式碼。**
