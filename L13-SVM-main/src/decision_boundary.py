# src/decision_boundary.py
"""
Utility functions for產生 meshgrid 並取得 SVM 的決策邊界或 decision function。

在 2D 教學圖中，我們需要將特徵空間離散化成格點，
再讓訓練好的 model 針對每個格點做預測，以得到 Z 值（類別或 decision function），
最後以 contour / colormap 顯示決策邊界。
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC


def create_meshgrid(X: np.ndarray, resolution: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """根據資料 X 建立 (xx, yy) meshgrid。

    参数說明:
    - X: shape (n_samples, 2)，僅支援 2 維特徵。
    - resolution: 每個維度的格點數目，預設 300（對應 300×300 的網格）。

    返回:
    - xx, yy: 兩個 shape (resolution, resolution) 的 2D 陣列，
      可直接與 model.predict/decision_function 搭配使用。
    """
    # 取得特徵的最小值與最大值，並在兩側各留 0.5 的 margin，以免邊緣裁切過緊
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    return xx, yy


def predict_decision_boundary(
    model: SVC,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """使用已訓練好的 SVC 直接預測每個格點的類別。

    這裡使用 `model.predict`，回傳的陣列會被 reshape 成與 xx/yy 相同的形狀，
    方便在 Matplotlib/Plotly 中以 `contourf` 繪製決策邊界。
    """
    # Flatten grid、合併成 (n_grid, 2) 再做預測
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    return Z.reshape(xx.shape)


def get_decision_function(
    model: SVC,
    xx: np.ndarray,
    yy: np.ndarray,
) -> np.ndarray:
    """取得模型的 decision_function（距離超平面的原始分數）。

    與 `predict_decision_boundary` 不同，此函式返回連續值，可用於繪製
    等高線（例如 margin、decision boundary、confidence region）。
    """
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid_points)
    return Z.reshape(xx.shape)

# ------------------------------------------------------------
# 範例用法（在 Jupyter/Streamlit 中可直接呼叫）
# ------------------------------------------------------------
# ```python
# xx, yy = create_meshgrid(X, resolution=300)
# Z_pred = predict_decision_boundary(model, xx, yy)
# Z_score = get_decision_function(model, xx, yy)
# ```
# ```
# 這段範例說明正好對應您在問題中提供的程式碼片段。
# ```
