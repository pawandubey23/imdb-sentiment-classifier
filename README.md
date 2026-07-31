# 🎬 IntelliReview — AI Movie Review Sentiment Analyzer

**Developed by Pawan Dubey**

An NLP project that classifies movie reviews as **Positive** or **Negative** using
TF-IDF + Logistic Regression, trained on the full **50,000-review IMDB Movie Reviews
dataset**, wrapped in a polished Streamlit UI.

---

## ✨ Features

- **Live sentiment analysis** — type or paste a review, get an instant prediction with confidence score
- **Explainability** — see the exact words that pushed the prediction toward Positive or Negative
- **Batch analysis** — upload a CSV of reviews and get predictions for all of them, downloadable as CSV
- **Session history** — every review analyzed is logged, viewable and downloadable
- **Word clouds** — the most common words across all positive vs. negative reviews in the dataset
- **Negation-aware** — correctly handles phrases like "not good" instead of only reading "good"

## 🧠 How it works

1. **Dataset** — 50,000 real IMDB reviews (25,000 positive / 25,000 negative) — the
   classic Stanford Large Movie Review Dataset (`IMDB_Dataset_50k.csv`).
2. **Cleaning** — HTML tags stripped, contractions expanded, negation words preserved
   and used to tag the words they negate (so "not good" ≠ "good" to the model).
3. **Split** — 75% train / 25% test, stratified (`random_state=42`).
4. **Vectorization** — TF-IDF, unigrams + bigrams, 30,000-word vocabulary.
5. **Model** — Logistic Regression (`scikit-learn`).
6. **Result** — ~90% accuracy on unseen test reviews.

## 📁 Project structure

```
sentiment_app/
├── app.py                     # Streamlit app (the UI)
├── train_model.py             # Trains the model, saves artifacts + word clouds
├── IMDB_Dataset_50k.csv       # Dataset (50,000 reviews)
├── vectorizer.joblib          # Saved TF-IDF vectorizer
├── sentiment_model.joblib     # Saved trained model
├── metrics.joblib             # Saved accuracy/F1/confusion matrix
├── positive_wordcloud.png     # Word cloud of positive reviews
├── negative_wordcloud.png     # Word cloud of negative reviews
├── requirements.txt
└── README.md
```

## 💻 Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Only needed once, or to retrain) Train the model
python train_model.py

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## ☁️ Deploy to Streamlit Community Cloud

1. **Push this folder to a new GitHub repo:**
   ```bash
   git init
   git add .
   git commit -m "IntelliReview: AI movie review sentiment analyzer"
   git branch -M main
   git remote add origin https://github.com/<your-username>/intellireview-sentiment-analysis.git
   git push -u origin main
   ```
   > `IMDB_Dataset_50k.csv` is ~64 MB and the two `.joblib` model files are small —
   > all well within GitHub's 100 MB per-file limit, so a normal `git push` works
   > (no Git LFS needed).

2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
3. Click **"New app"**, select your repo, branch `main`, and set the main file to `app.py`.
4. Click **Deploy**. Streamlit Cloud will install `requirements.txt` and launch the app —
   your model files are already committed, so it won't need to retrain on startup.
5. Once live, copy your app's URL (e.g. `https://intellireview-sentiment-analysis.streamlit.app`)
   and add it to your GitHub repo description, resume, and LinkedIn post.

## 🛠️ Tech stack

Python · Pandas · Scikit-learn (TF-IDF + Logistic Regression) · Streamlit · Plotly

---

*Built as part of an AI/ML portfolio project — Developed by **Pawan Dubey**.*