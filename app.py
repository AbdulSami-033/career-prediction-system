import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Career Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
data = pd.read_csv("CareerMapping1.csv")
data.drop(columns=["Unnamed: 0"], inplace=True)

y = data.pop("Role")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

label_map = dict(enumerate(label_encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    data, y_encoded, test_size=0.3, random_state=42
)

# ---------------- TRAIN KNN MODEL ----------------
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

# ---------------- UI HEADER ----------------
st.title("🎓 Career Prediction System")
st.markdown("**Powered by K-Nearest Neighbors (KNN) Algorithm**")

st.info(
    "Enter your skill levels honestly (0–100). "
    "The model will predict the most suitable career for you."
)

# ---------------- USER INPUT ----------------
st.subheader("📥 Enter Your Skills")

user_input = []

col1, col2 = st.columns(2)

for i, feature in enumerate(data.columns):
    if i % 2 == 0:
        value = col1.slider(feature, 0, 100, 50)
    else:
        value = col2.slider(feature, 0, 100, 50)
    user_input.append(value)

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Career"):
    user_input = np.array(user_input).reshape(1, -1)

    prediction = knn.predict(user_input)[0]
    probabilities = knn.predict_proba(user_input)[0]

    top_3 = np.argsort(probabilities)[-3:][::-1]

    st.success(f"### ✅ Best Career Match: **{label_map[prediction]}**")

    st.subheader("📊 Top 3 Career Recommendations")

    for idx in top_3:
        st.write(f"**{label_map[idx]}** — {probabilities[idx] * 100:.2f}%")
        st.progress(int(probabilities[idx] * 100))
