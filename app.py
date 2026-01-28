import streamlit as st

# MUST BE FIRST
st.set_page_config(page_title="Personality AI", layout="centered")

import re
import numpy as np
import os

# Imports WITHOUT Streamlit calls
try:
    from tensorflow.keras.models import load_model
except ImportError:
    st.error("TensorFlow not installed. Run: pip install tensorflow-cpu")
    st.stop()

try:
    import joblib
except ImportError:
    st.error("Joblib not installed. Run: pip install joblib")
    st.stop()

st.title("Personality Prediction")

# Check files
required_files = [
    "personality_model.keras",
    "tfidf_vectorizer.pkl",
    "label_scaler.pkl"
]

missing = [f for f in required_files if not os.path.exists(f)]

if missing:
    st.error(f"Missing files: {', '.join(missing)}")
    st.stop()

# Load model
with st.spinner("Loading model..."):
    model = load_model("personality_model.keras")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    label_scaler = joblib.load("label_scaler.pkl")

traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

text = st.text_area("Enter text", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned]).toarray()
        pred = model.predict(vec)
        pred = label_scaler.inverse_transform(pred)[0]

        st.subheader("🧠 Personality Prediction Results")

        for t, v in zip(traits, pred):
            v = float(v)
            st.write(f"**{t}: {v:.2f}**")
            st.progress(min(max(v / 5.0, 0), 1))
