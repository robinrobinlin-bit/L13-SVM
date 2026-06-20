# src/svm_model.py
"""SVM 模型相關函式，封裝 sklearn.svm.SVC 的訓練與評估。"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_svm_model(X, y, kernel="linear", C=1.0, gamma="scale", degree=3):
    """使用 sklearn 訓練 SVM 模型。
    參數說明：
    - X, y: 訓練資料
    - kernel: linear / rbf / poly / sigmoid
    - C: 正則化參數
    - gamma: RBF/Poly/Sigmoid 的 kernel coefficient
    - degree: 多項式核的次數 (僅在 poly 時使用)
    回傳 (model, accuracy)
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, probability=False)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    return model, acc


def get_support_vectors(model):
    """取得模型的 support vectors (numpy 陣列)。"""
    return model.support_vectors_


def calculate_accuracy(model, X, y):
    """計算模型在給定資料上的準確率。"""
    pred = model.predict(X)
    return accuracy_score(y, pred)
