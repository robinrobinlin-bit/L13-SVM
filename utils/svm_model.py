# src/svm_model.py
"""
SVM Model utilities for the SVM 教學專案.

提供以下功能:
- 訓練 SVM 模型 (支援自訂 kernel、C、gamma、degree)
- 取得支援向量
- 計算模型在測試資料上的準確度

所有函式均加入中文說明與型別提示，方便在 Streamlit 中直接呼叫。
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_svm(
    X,
    y,
    kernel="linear",
    C=1.0,
    gamma="scale",
    degree=3,
):
    """訓練一個 Support Vector Machine 模型。

    參數說明:
    - X: 特徵矩陣，形狀 (n_samples, n_features)
    - y: 標籤向量，形狀 (n_samples,)
    - kernel: 核函式類型，支援 "linear", "rbf", "poly", "sigmoid"
    - C: 正則化參數，C 越大模型越不容許錯誤分類
    - gamma: 核函式的 gamma 參數，對於 "rbf"、"poly"、"sigmoid" 有效；
      亦可使用 "scale" 或 "auto"
    - degree: 多項式核的階數 (僅在 kernel='poly' 時使用)

    回傳:
    - 訓練好的 sklearn.svm.SVC 物件
    """
    # 建立 SVC 物件，所有參數皆直接傳入 sklearn
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=False)
    model.fit(X, y)
    accuracy = model.score(X, y)
    return model, accuracy

def get_support_vectors(model: SVC) -> np.ndarray:
    """取得訓練好的模型之 support vectors。

    返回形狀為 (n_support_vectors, n_features) 的 numpy 陣列。
    """
    return model.support_vectors_


def calculate_accuracy(model: SVC, X: np.ndarray, y: np.ndarray) -> float:
    """計算模型在給定測試資料上的分類準確率。"""
    y_pred = model.predict(X)
    return float(accuracy_score(y, y_pred))

def get_model(X, y, kernel, C, gamma, degree):
    """Thin wrapper returning model and training accuracy.
    Mirrors the original get_model API used in pages.
    """
    return train_svm(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)
