"""Train a TF-IDF + Logistic Regression sentiment classifier and evaluate it on held-out hand-written sentences."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from data import TRAIN_TEXTS, TRAIN_LABELS, TEST_TEXTS, TEST_LABELS

def main():
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=1000),
    )
    model.fit(TRAIN_TEXTS, TRAIN_LABELS)

    preds = model.predict(TEST_TEXTS)
    print(f"Accuracy on {len(TEST_TEXTS)} unseen hand-written sentences: {accuracy_score(TEST_LABELS, preds):.3f}")
    print(classification_report(TEST_LABELS, preds, target_names=["negative", "positive"]))

    joblib.dump(model, "sentiment_model.joblib")
    print("Saved model to sentiment_model.joblib")

if __name__ == "__main__":
    main()
