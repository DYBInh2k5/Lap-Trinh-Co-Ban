import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocess import clean_text


def load_and_prepare(path="data/data.csv"):
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    return df["text_clean"], df["label"]


def main():
    X, y = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2],
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__class_weight": [None, "balanced"]
    }

    gs = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1, verbose=0)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    out_dir = "artifacts_improved"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model_improved.pkl")
    joblib.dump(best, model_path)

    metrics_path = os.path.join(out_dir, "metrics.json")
    metrics = {"accuracy": acc, "report": report, "best_params": gs.best_params_}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Improved training done.")
    print(f"Best params: {gs.best_params_}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved model to {model_path} and metrics to {metrics_path}")


if __name__ == "__main__":
    main()
