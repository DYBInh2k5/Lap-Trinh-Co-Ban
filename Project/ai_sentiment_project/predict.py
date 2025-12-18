import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_text(text):
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred

if __name__ == "__main__":
    while True:
        text = input("Nhập câu (gõ 'exit' để thoát): ")
        if text.lower() in ("exit", "quit"):
            break
        print("Cảm xúc:", predict_text(text))
