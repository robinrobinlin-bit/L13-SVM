# pages/0_Theory.py
"""
SVM 基本概念說明頁面（理論篇）

本頁面以問答形式解釋 SVM 為何重要、核心概念與 sklearn SVC 的運作。
在 Streamlit 側欄可透過 "Theory" 或從首頁連結進入。
"""

import streamlit as st

st.title("🔎 SVM 理論概念」")

questions = [
    "**1️⃣ 為什麼需要 SVM？**",
    "**2️⃣ 什麼是 margin？**",
    "**3️⃣ 什麼是 support vector？**",
    "**4️⃣ 為什麼 linear 不夠？**",
    "**5️⃣ kernel 如何幫助非線性分類？**",
    "**6️⃣ sklearn 的 SVC 如何真正訓練模型？**",
    "**7️⃣ C、gamma、kernel 改變後，決策面如何變化？**",
]

answers = [
    "SVM（Support Vector Machine）是一種監督式機器學習演算法，\n" 
    "它在二元分類問題上能取得**最大化 margin**的決策邊界，\n" 
    "有較好的泛化能力，特別適合樣本量不大且特徵維度較高的情境。",
    "**Margin**指的是決策平面（hyperplane）與最近資料點（support vectors）之間的距離。\n" 
    "SVM 透過最大化這個間隔，使模型在未見資料上更具魯棒性。",
    "**Support Vector** 是距離決策平面最近的資料點，這些點決定了最終的分隔平面。\n" 
    "只有 support vectors 影響模型參數，其餘遠離平面的點對模型影響極小。",
    "線性 SVM 只能在特徵空間中劃出一條直線（2D）或超平面（高維），\n" 
    "當資料本身非線性可分時（例如同心圓或 moons），單純的線性分割無法成功。",
    "**Kernel Trick** 透過核函式將資料映射到更高維度的特徵空間，\n" 
    "在該空間裡資料往往變得線性可分，SVM 仍然只在映射後的空間中尋找線性平面，\n" 
    "而不需要顯式計算高維特徵向量。常見的核函式有 `linear`、`rbf`、`poly`、`sigmoid`。",
    "`sklearn.svm.SVC` 內部會先根據選擇的 kernel 計算 **核矩陣**（每對樣本的相似度），\n" 
    "再以二次規劃（Quadratic Programming）求解拉格朗日乘子 \(\alpha\)。\n" 
    "支援向量對應的 \(\alpha > 0\) ，最終模型由這些向量決定。",
    "- **C**（正則化參數）調大會減少容許錯誤，決策面更貼合訓練資料，易過擬合；\n" 
    "  調小會放寬錯誤容忍，決策面更平滑，margin 較寬。\n" 
    "- **gamma**（RBF、poly、sigmoid 核的參數）值大時單一點影響範圍小，決策面變得更彎曲、易過擬合；\n" 
    "  值小則影響範圍大，決策面較平滑。\n" 
    "- **kernel** 的選擇決定了模型在特徵空間的形狀：\n" 
    "  `linear` → 真正的線性平面；\n" 
    "  `rbf` → 隱含無限維特徵，能擬合複雜非線性邊界；\n" 
    "  `poly` → 多項式特徵，控制 degree 可調整曲線的複雜度。"
]

for q, a in zip(questions, answers):
    st.markdown(q)
    st.write(a)
    st.divider()

st.info("在左側導覽列可切換至互動示範章節，透過滑桿即時觀察 C、gamma、kernel 對決策面的影響。")
