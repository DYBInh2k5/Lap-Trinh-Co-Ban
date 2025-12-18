import re
import unicodedata

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    s = normalize_unicode(s)
    s = s.lower()
    # remove urls
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    # remove emails
    s = re.sub(r"\S+@\S+", " ", s)
    # remove punctuation (keep unicode letters and numbers and spaces)
    s = re.sub(r"[^\w\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠ-ỹỳ̉̃̉̀́]+", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

if __name__ == "__main__":
    samples = [
        "I love this product!",
        "Giao hàng nhanh, đóng gói cẩn thận",
        "Check https://example.com now",
    ]
    for t in samples:
        print(t, "->", clean_text(t))
