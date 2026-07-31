"""
Sentiment Analysis - Model Training Script
Developed by Pawan Dubey

Follows the exact pipeline taught in "AI Playground: Project 1 - Sentiment Analysis":
  1. Load the dataset
  2. Explore it (shape, class balance)
  3. Clean the text
  4. Train / test split (75/25, stratified, random_state=42 - same as the notebook)
  5. Convert text to numbers with TF-IDF (lowercase, English stopwords - same as the notebook)
  6. Train a Logistic Regression classifier (random_state=42 - same as the notebook)
  7. Evaluate: accuracy, classification report, confusion matrix
  8. Save the trained vectorizer + model so the Streamlit app can load them instantly

Dataset used: the full 50,000-review IMDB Movie Reviews dataset (Maas et al., Stanford AI Lab)
instead of the notebook's original 232-row hand-built demo dataset, per request for a
larger, real-world dataset. Same TF-IDF + Logistic Regression approach as the notebook.
"""

import re
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

DATA_PATH = "IMDB_Dataset_50k.csv"

print("=" * 60)
print("STEP 1: Load the dataset")
print("=" * 60)
sentiment_df = pd.read_csv(DATA_PATH)
sentiment_df.columns = ["review", "sentiment"]
sentiment_df["sentiment"] = sentiment_df["sentiment"].map({"positive": 1, "negative": 0})
sentiment_df = sentiment_df.dropna().drop_duplicates(subset="review").reset_index(drop=True)
print("Dataset shape:", sentiment_df.shape)
print("\nClass distribution:")
print(sentiment_df["sentiment"].value_counts())

print("\n" + "=" * 60)
print("STEP 2: Clean the text (remove HTML tags, extra whitespace)")
print("=" * 60)


CONTRACTIONS = {
    "isn't": "is not", "wasn't": "was not", "aren't": "are not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "can not", "couldn't": "could not", "won't": "will not",
    "wouldn't": "would not", "shouldn't": "should not", "mustn't": "must not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
}
NEGATION_WORDS = {"not", "no", "never", "cannot", "none", "nobody", "nothing", "neither", "nor"}
CONTRAST_WORDS = {"but", "however", "though", "although", "yet", "except"}

def clean_review(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)

    tokens = re.findall(r"[a-z']+|[.,!?;]", text)
    output, negate = [], False
    for tok in tokens:
        if tok in ".,!?;":
            negate = False
            continue
        if tok in CONTRAST_WORDS:
            negate = False
            output.append(tok)
            continue
        if tok in NEGATION_WORDS:
            output.append(tok)
            negate = True
            continue
        output.append(f"not_{tok}" if negate else tok)

    cleaned = " ".join(output)
    cleaned = re.sub(r"[^a-z_\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


sentiment_df["review"] = sentiment_df["review"].apply(clean_review)
print("Sample cleaned review:\n", sentiment_df["review"].iloc[0][:200], "...")

print("\n" + "=" * 60)
print("STEP 3: Train / test split (75% / 25%, stratified)")
print("=" * 60)
X_text = sentiment_df["review"]
y_sentiment = sentiment_df["sentiment"]

X_train_text, X_test_text, y_train_sentiment, y_test_sentiment = train_test_split(
    X_text, y_sentiment,
    test_size=0.25,
    random_state=42,
    stratify=y_sentiment,
)
print("Training samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))

print("\n" + "=" * 60)
print("STEP 4: Convert text to numbers using TF-IDF")
print("=" * 60)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stop_words = ENGLISH_STOP_WORDS - NEGATION_WORDS  # keep "not", "no", "never", etc.

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=list(custom_stop_words),
    max_features=30000,
    ngram_range=(1, 2),
    min_df=3,
)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)
print("Training matrix shape:", X_train_tfidf.shape)
print("Testing matrix shape:", X_test_tfidf.shape)

print("\n" + "=" * 60)
print("STEP 5: Train the Logistic Regression sentiment model")
print("=" * 60)
start = time.time()
sentiment_model = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
sentiment_model.fit(X_train_tfidf, y_train_sentiment)
print(f"Model trained in {time.time() - start:.1f} seconds")

print("\n" + "=" * 60)
print("STEP 6: Evaluate the model")
print("=" * 60)
sentiment_predictions = sentiment_model.predict(X_test_tfidf)
sentiment_accuracy = accuracy_score(y_test_sentiment, sentiment_predictions)
sentiment_f1 = f1_score(y_test_sentiment, sentiment_predictions)

print("Accuracy:", round(sentiment_accuracy, 4))
print("F1-score:", round(sentiment_f1, 4))
print("\nClassification Report:")
print(classification_report(y_test_sentiment, sentiment_predictions, zero_division=0))

cm = confusion_matrix(y_test_sentiment, sentiment_predictions)
print("Confusion Matrix:\n", cm)

print("\n" + "=" * 60)
print("STEP 7: Save the trained vectorizer + model + metrics")
print("=" * 60)
joblib.dump(vectorizer, "vectorizer.joblib")
joblib.dump(sentiment_model, "sentiment_model.joblib")

metrics = {
    "accuracy": float(sentiment_accuracy),
    "f1_score": float(sentiment_f1),
    "confusion_matrix": cm.tolist(),
    "train_size": len(X_train_text),
    "test_size": len(X_test_text),
    "total_reviews": len(sentiment_df),
    "vocab_size": len(vectorizer.vocabulary_),
}
joblib.dump(metrics, "metrics.joblib")

print("Saved: vectorizer.joblib, sentiment_model.joblib, metrics.joblib")
print("\nAll done.")
