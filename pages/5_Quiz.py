import streamlit as st

st.title("SVM 教學互動網站 - 第 5 章：小測驗")

questions = [
    {
        "question": "1. Support Vectors 是什麼？",
        "options": [
            "離決策邊界最近、決定邊界位置的資料點",
            "所有訓練資料中最遠的點",
            "隨機挑選的樣本",
            "模型的參數向量"
        ],
        "answer": 0,
        "explanation": "Support Vectors 是最靠近決策平面的資料點，它們決定了 margin，移除它們會改變超平面。"
    },
    {
        "question": "2. C 變大通常會造成什麼效果？",
        "options": [
            "模型對錯誤分類容忍度降低，可能 overfit",
            "模型對錯誤分類容忍度提升，可能 underfit",
            "模型不會受到影響",
            "模型會自動選擇不同的 kernel"
        ],
        "answer": 0,
        "explanation": "較大的 C 會強迫模型盡量不犯錯誤分類，導致決策邊界更貼合訓練資料，易發生 overfitting。"
    },
    {
        "question": "3. gamma 變大通常會造成什麼效果？",
        "options": [
            "每個樣本影響範圍縮小，決策邊界變得更彎曲，易 overfit",
            "每個樣本影響範圍擴大，決策邊界變得更平滑",
            "不會影響模型表現",
            "會改變資料的維度"
        ],
        "answer": 0,
        "explanation": "在 RBF 或 poly kernel 中，較大的 gamma 使核函數局部化，模型會捕捉更細緻的模式，容易 overfit。"
    },
    {
        "question": "4. margin 越大代表什麼？",
        "options": [
            "分類平面與最近樣本的距離較遠，模型較穩健",
            "模型對錯誤分類容忍度高，容易 underfit",
            "模型的訓練時間更長",
            "支援向量的數量會變少"
        ],
        "answer": 0,
        "explanation": "較大的 margin 表示兩類最近的樣本之間距離較遠，模型在未見資料上較具廣泛的泛化能力。"
    },
    {
        "question": "5. RBF kernel 適合處理什麼資料？",
        "options": [
            "非線性、形狀呈彎曲或同心圓的資料",
            "純線性可分的資料",
            "高維稀疏的文字向量",
            "只有單一特徵的資料"
        ],
        "answer": 0,
        "explanation": "RBF kernel 透過高斯函數把資料映射到無限維空間，特別適合處理非線性、曲線或圓形分布的資料。"
    },
]

score = 0
for q in questions:
    st.subheader(q["question"])
    user_choice = st.radio("選項", q["options"], key=q["question"])
    if st.button("提交答案", key=q["question"] + "_submit"):
        selected_index = q["options"].index(user_choice)
        if selected_index == q["answer"]:
            st.success("答對了！ 🎉")
            score += 1
        else:
            st.error("答錯了。")
        st.write("**解釋:**", q["explanation"])
        st.markdown("---")

st.write(f"### 最終得分: {score} / {len(questions)}")
