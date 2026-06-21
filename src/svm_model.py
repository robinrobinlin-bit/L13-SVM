# -*- coding: utf-8 -*-
"""src/svm_model.py
提供 SVM 訓練與相關工具函式。
所有函式均返回 Python 原生類型，便於在 Streamlit 中直接使用。
"""

from __future__ import annotations

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from typing import Tuple


def train_svm(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "linear",
    C: float = 1.0,
    gamma: str | float = "scale",
    degree: int = 3,
) -> Tuple[SVC, float]:
    """訓練 Support Vector Machine。

    Parameters
    ----------
    X, y : np.ndarray
        特徵矩陣與標籤向量。
    kernel : str, default "linear"
        核函式，支援 "linear", "rbf", "poly" 等。
    C : float, default 1.0
        正則化參數，值越大模型越不容忍錯誤分類。
    gamma : str | float, default "scale"
        核函式的 gamma 參數，對於非 linear 有效。
    degree : int, default 3
        多項式核的階數（僅在 kernel='poly' 時使用）。

    Returns
    -------
    model : sklearn.svm.SVC
        訓練好的模型。
    accuracy : float
        訓練資料上的準確率。
    """
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=False,
    )
    model.fit(X, y)
    accuracy = model.score(X, y)
    return model, accuracy


def get_model(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str,
    C: float,
    gamma: str | float,
    degree: int,
) -> Tuple[SVC, float]:
    """Thin wrapper，保持與舊版 API 相容。
    直接呼叫 ``train_svm``，返回模型與訓練準確度。
    """
    return train_svm(X, y, kernel=kernel, C=C, gamma=gamma, degree=degree)


def calculate_accuracy(model: SVC, X: np.ndarray, y: np.ndarray) -> float:
    """計算模型在測試資料上的分類準確率。"""
    y_pred = model.predict(X)
    return float(accuracy_score(y, y_pred))


def get_support_vectors(model: SVC) -> np.ndarray:
    """取得支援向量座標（2D numpy 陣列）。"""
    return model.support_vectors_
