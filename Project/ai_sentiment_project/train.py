import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


def main():
    data_path = os.path.join("data", "data.csv")
    df = pd.read_csv(data_path)
    X = df["text"].fillna("")
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    model_path = os.path.join(artifacts_dir, "model.pkl")
    vect_path = os.path.join(artifacts_dir, "vectorizer.pkl")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vect_path)

    metrics = {"accuracy": acc, "report": report}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Training finished. Saved {model_path} and {vect_path}")
    print(f"Accuracy on test set: {acc:.4f}")


if __name__ == "__main__":
    main()
