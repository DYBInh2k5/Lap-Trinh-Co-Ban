from transformers import pipeline

def main():
    chatbot = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
    )

    print("Gõ 'exit' để thoát.")
    while True:
        text = input("You: ")
        if text.lower() in ("exit", "quit"):
            break
        results = chatbot(text)
        label = results[0]["label"]
        score = results[0]["score"]
        mapping = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
        readable = mapping.get(label, label)
        print(f"AI: {readable} (score: {score:.3f})")

if __name__ == "__main__":
    main()
