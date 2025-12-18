# AI Sentiment Analysis Project

## Mô tả
Dự án phân loại cảm xúc văn bản (Tiếng Anh / Tiếng Việt) sử dụng TF-IDF + Logistic Regression và chatbot real-time với mô hình Transformer.

## Cấu trúc

ai_sentiment_project/
│
├── data/
│   └── data.csv
├── train.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md

## Cài đặt

```
pip install -r requirements.txt
```

## Chạy

1. Train model (lưu `model.pkl` và `vectorizer.pkl`):

```
python train.py
```

2. Dự đoán qua console (dùng model đã train):

```
python predict.py
```

3. Chatbot dùng Transformer (yêu cầu tải mô hình từ HuggingFace):

```
python app.py
```

4. Web app bằng Streamlit (sau khi chạy `python train.py` để tạo `artifacts/`):

```
streamlit run app_streamlit.py
```

## Ghi chú
- `train.py` giờ chia train/test, in accuracy và lưu model/vectorizer vào `artifacts/`.
- `app_streamlit.py` dùng các file trong `artifacts/` để dự đoán.

## Improved training

- `train_improved.py` applies basic text cleaning, TF-IDF with n-grams and `LogisticRegression`.
- It runs a small `GridSearchCV` and saves the best pipeline to `artifacts_improved/` along with `metrics.json`.

Run improved training:

```
python train_improved.py
```

## Ghi vào CV (ví dụ)

AI Sentiment Analysis Project — Built an NLP system to classify text sentiment using TF-IDF, Logistic Regression and HuggingFace Transformers. Developed interactive chatbot for real-time sentiment prediction.
