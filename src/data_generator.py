# -*- coding: utf-8 -*-
"""src/data_generator.py
產生合成資料集的工具函式。
支援 `blobs`、`moons`、`circles`、`classification` 四種資料型別。
返回 (X, y, meta) 其中 meta 包含資料類型與 random_state。
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from typing import Tuple, Dict


def make_dataset(
    kind: str,
    n_samples: int = 250,
    noise: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """產生合成資料。

    Parameters
    ----------
    kind: str
        資料類型，支援 'blobs', 'moons', 'circles', 'classification'.
    n_samples: int, optional
        樣本數，預設 250。
    noise: float, optional
        雜訊程度，預設 0.1。
    random_state: int, optional
        隨機種子，預設 42。

    Returns
    -------
    X: np.ndarray
        特徵矩陣，形狀 (n_samples, 2)。
    y: np.ndarray
        標籤向量，形狀 (n_samples,)。
    meta: dict
        包含 'kind' 與 'random_state' 的資訊，供後續追蹤使用。
    """
    # 依照類型呼叫對應的 sklearn 產生函式
    if kind == "blobs":
        result = make_blobs(
            n_samples=n_samples,
            centers=2,
            cluster_std=1.0,
            random_state=random_state,
        )
        X = result[0]
        y = result[1]
    elif kind == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif kind == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    elif kind == "classification":
        # 使用 sklearn 的 make_classification 作為一般分類資料
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            class_sep=1.0,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    meta = {"kind": kind, "random_state": random_state}
    return X, y, meta
