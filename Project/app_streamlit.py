import os
import joblib
import streamlit as st

st.title("AI Sentiment Analysis")
st.write("Enter text (English or Vietnamese) and click Predict.")

artifacts_dir = "artifacts"
model_path = os.path.join(artifacts_dir, "model.pkl")
vect_path = os.path.join(artifacts_dir, "vectorizer.pkl")

if not (os.path.exists(model_path) and os.path.exists(vect_path)):
    st.warning("Models not found. Run `python train.py` first to create artifacts/ model files.")
else:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    text = st.text_area("Input text", height=120)
    if st.button("Predict"):
        if not text.strip():
            st.info("Please enter some text.")
        else:
            X = vectorizer.transform([text])
            try:
                proba = model.predict_proba(X)[0]
                idx = proba.argmax()
                label = model.classes_[idx]
                score = proba[idx]
                st.success(f"Prediction: {label} (score: {score:.3f})")
            except Exception:
                pred = model.predict(X)[0]
                st.success(f"Prediction: {pred}")

    st.markdown("---")
    st.write("Model files in artifacts/:")
    for f in os.listdir(artifacts_dir):
        st.write(f)
