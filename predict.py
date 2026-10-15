"""Predict sentiment. Usage:  python predict.py "your sentence"   (or run with no args for interactive mode)"""
import sys
import joblib

model = joblib.load("sentiment_model.joblib")

def classify(text):
    p = model.predict_proba([text])[0][1]
    return ("POSITIVE" if p >= 0.5 else "NEGATIVE"), p

if len(sys.argv) > 1:
    label, p = classify(" ".join(sys.argv[1:]))
    print(f"{label} (positive probability {p:.2f})")
else:
    print("Type a sentence (empty line to quit).")
    while True:
        text = input("> ").strip()
        if not text:
            break
        label, p = classify(text)
        print(f"  {label} (positive probability {p:.2f})")
