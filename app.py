"""
🎬 IntelliReview — AI Movie Review Sentiment Analyzer
Developed by Pawan Dubey

A TF-IDF + Logistic Regression sentiment classifier trained on the full
50,000-review IMDB Movie Reviews dataset, wrapped in a Streamlit UI.
"""

import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliReview | AI Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #f3ecff 0%, #eef2ff 45%, #f7f9ff 100%);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #d6336c, #6d28d9, #059669);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #4b4768;
        font-size: 1.02rem;
        max-width: 640px;
        margin: 0 auto;
    }

    .metric-box {
        text-align: center;
        background: #ffffff;
        border: 1px solid #e6e1f7;
        border-radius: 14px;
        padding: 0.9rem 0.4rem;
        box-shadow: 0 2px 10px rgba(109, 40, 217, 0.06);
    }
    .metric-box .val { font-size: 1.5rem; font-weight: 700; color: #241f47; }
    .metric-box .lbl { font-size: 0.78rem; color: #726d94; text-transform: uppercase; letter-spacing: .04em;}

    .result-positive {
        background: linear-gradient(135deg, rgba(34,197,94,0.14), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.35);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(239,68,68,0.14), rgba(239,68,68,0.03));
        border: 1px solid rgba(239,68,68,0.35);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
    }
    .result-emoji { font-size: 3.2rem; }
    .result-label { font-size: 1.6rem; font-weight: 700; color: #201c3a; margin-top: 0.2rem;}
    .result-conf { color: #55507a; font-size: 0.95rem; margin-top: 0.3rem;}

    .stTextArea textarea {
        background: #ffffff !important;
        border: 1.5px solid #ddd6f3 !important;
        border-radius: 12px !important;
        color: #1f1b3a !important;
        font-size: 1rem !important;
        caret-color: #6d28d9 !important;
    }
    .stTextArea textarea::placeholder { color: #9992b8 !important; }

    div.stButton > button {
        background: linear-gradient(90deg, #6d28d9, #d6336c);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.15s ease;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.25);
    }
    div.stButton > button:hover { transform: translateY(-2px); }

    .footer-credit {
        text-align: center;
        color: #746f97;
        font-size: 0.85rem;
        padding: 1.6rem 0 0.6rem 0;
        border-top: 1px solid #e6e1f7;
        margin-top: 2rem;
    }
    .footer-credit b { color: #3d3768; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load model artifacts (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("vectorizer.joblib")
    model = joblib.load("sentiment_model.joblib")
    metrics = joblib.load("metrics.joblib")
    return vectorizer, model, metrics


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
    """Must exactly mirror the cleaning used in train_model.py, or the
    vectorizer's vocabulary won't line up with what the model was trained on."""
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


def predict_sentiment(review, vectorizer, model):
    cleaned = clean_review(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]
    confidence = proba.max()
    return prediction, confidence, proba


def get_influential_words(review, vectorizer, model, top_n=6):
    """Explainability: shows which specific words in the review pushed the
    model's decision toward Positive or Negative, using the model's own
    learned weight for each word combined with how strongly that word
    appears in this particular review (its TF-IDF value)."""
    cleaned = clean_review(review)
    vector = vectorizer.transform([cleaned])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]

    nonzero_idx = vector.nonzero()[1]
    contributions = [(feature_names[i], vector[0, i] * coefs[i]) for i in nonzero_idx]
    contributions.sort(key=lambda x: x[1], reverse=True)

    top_positive = [c for c in contributions if c[1] > 0][:top_n]
    top_negative = [c for c in contributions if c[1] < 0][-top_n:]
    return top_positive, top_negative


if "history" not in st.session_state:
    st.session_state.history = []


try:
    vectorizer, model, metrics = load_artifacts()
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 IntelliReview")
    st.caption("AI-powered movie review sentiment analyzer")
    st.markdown("---")

    if artifacts_ready:
        st.markdown("**📊 Model Snapshot**")
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("F1-score", f"{metrics['f1_score']*100:.2f}%")
        st.metric("Training reviews", f"{metrics['train_size']:,}")
        st.metric("Test reviews", f"{metrics['test_size']:,}")
        st.markdown("---")

    st.markdown("**⚙️ How it works**")
    st.write(
        "1. Text is cleaned (HTML/punctuation removed)\n"
        "2. Converted to numbers via **TF-IDF**\n"
        "3. Classified by **Logistic Regression**\n"
        "4. Trained on **50,000 real IMDB reviews**"
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem;color:#8b86ab;'>Built with Python, "
        "Scikit-learn, Pandas &amp; Streamlit</div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Hero
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 IntelliReview</h1>
    <p>Type any movie review below and watch the AI decide — in real time — whether it's Positive or Negative, trained on 50,000 real IMDB reviews.</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_ready:
    st.error(
        "Model files not found. Please run `python train_model.py` first to "
        "generate `vectorizer.joblib`, `sentiment_model.joblib`, and `metrics.joblib`."
    )
    st.stop()

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab1, tab_batch, tab_history, tab2, tab_clouds, tab3 = st.tabs([
    "🔮 Analyze a Review",
    "📦 Batch Analysis",
    "📜 History",
    "📊 Model Performance",
    "🔍 Word Clouds",
    "ℹ️ About the Project",
])

# ---------------- TAB 1: Analyze ----------------
with tab1:
    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        with st.container(border=True):
            st.markdown("#### ✍️ Write or paste a movie review")

            example_reviews = {
                "😍 Glowing praise": "This movie was an absolute masterpiece — the acting, the direction, everything about it was breathtaking.",
                "😴 Total letdown": "I was so bored throughout the entire film, the plot made no sense and the acting felt lifeless.",
                "🎭 Mixed feelings": "The visuals were stunning but the story dragged on and the ending felt rushed.",
            }

            chosen_example = st.selectbox(
                "Try an example, or write your own below:",
                ["-- Write my own --"] + list(example_reviews.keys()),
            )

            default_text = "" if chosen_example == "-- Write my own --" else example_reviews[chosen_example]

            user_review = st.text_area(
                "Your review",
                value=default_text,
                height=180,
                placeholder="e.g. The cinematography was gorgeous but the pacing really let the film down...",
                label_visibility="collapsed",
            )

            analyze_clicked = st.button("✨ Analyze Sentiment", use_container_width=True)

    with col_result:
        with st.container(border=True):
            st.markdown("#### 🎯 Result")

            if analyze_clicked and user_review.strip():
                prediction, confidence, proba = predict_sentiment(user_review, vectorizer, model)

                st.session_state.history.insert(0, {
                    "time": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "review": user_review.strip()[:120] + ("..." if len(user_review.strip()) > 120 else ""),
                    "sentiment": "Positive" if prediction == 1 else "Negative",
                    "confidence": f"{confidence*100:.1f}%",
                })

                if prediction == 1:
                    st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-emoji">😊</div>
                        <div class="result-label">Positive</div>
                        <div class="result-conf">{confidence*100:.1f}% confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-emoji">😞</div>
                        <div class="result-label">Negative</div>
                        <div class="result-conf">{confidence*100:.1f}% confidence</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                fig = go.Figure(go.Bar(
                    x=[proba[0]*100, proba[1]*100],
                    y=["Negative", "Positive"],
                    orientation="h",
                    marker_color=["#f87171", "#4ade80"],
                    text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                    textposition="outside",
                ))
                fig.update_layout(
                    height=180,
                    margin=dict(l=10, r=30, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#241f47"),
                    xaxis=dict(range=[0, 100], showgrid=False, title="Probability (%)"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)

                top_pos, top_neg = get_influential_words(user_review, vectorizer, model)
                if top_pos or top_neg:
                    st.markdown("**🔬 Key words that drove this decision**")
                    words = [w.replace("not_", "not ") for w, _ in top_neg] + [w.replace("not_", "not ") for w, _ in top_pos]
                    values = [v for _, v in top_neg] + [v for _, v in top_pos]
                    colors = ["#f87171"] * len(top_neg) + ["#4ade80"] * len(top_pos)
                    fig_words = go.Figure(go.Bar(
                        x=values, y=words, orientation="h", marker_color=colors,
                    ))
                    fig_words.update_layout(
                        height=max(180, 32 * len(words)),
                        margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#241f47", size=12),
                        xaxis=dict(showgrid=False, title="Push toward Negative ← → Positive"),
                        yaxis=dict(showgrid=False),
                    )
                    st.plotly_chart(fig_words, use_container_width=True)

            elif analyze_clicked:
                st.warning("Please type a review first.")
            else:
                st.info("Your result will appear here after you click **Analyze Sentiment**.")

# ---------------- TAB 2: Model Performance ----------------
with tab2:
    with st.container(border=True):
        st.markdown("#### 📈 How well does the model actually perform?")
        st.write(
            "The model is trained on 75% of the dataset and evaluated on the remaining 25% — "
            "reviews it has never seen before. This mirrors real-world deployment conditions."
        )

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip(
            [c1, c2, c3, c4],
            [
                ("Accuracy", f"{metrics['accuracy']*100:.2f}%"),
                ("F1-score", f"{metrics['f1_score']*100:.2f}%"),
                ("Vocabulary size", f"{metrics['vocab_size']:,}"),
                ("Total reviews", f"{metrics['total_reviews']:,}"),
            ],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="val">{val}</div>
                    <div class="lbl">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cm = np.array(metrics["confusion_matrix"])
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Purples",
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Negative", "Positive"],
            y=["Negative", "Positive"],
        )
        fig_cm.update_layout(
            title="Confusion Matrix (on held-out test reviews)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#241f47"),
            height=420,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------------- TAB: Batch Analysis ----------------
with tab_batch:
    with st.container(border=True):
        st.markdown("#### 📦 Batch Analysis")
        st.write(
            "Upload a CSV file with a column of movie reviews to get predictions for "
            "all of them at once — useful for analyzing a whole dataset of reviews in one go."
        )

        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Could not read that file as a CSV: {e}")
                batch_df = None

            if batch_df is not None and len(batch_df) > 0:
                text_columns = [c for c in batch_df.columns if batch_df[c].dtype == object]
                if not text_columns:
                    st.warning("No text columns found in this file.")
                else:
                    review_col = st.selectbox("Which column contains the review text?", text_columns)
                    run_batch = st.button("🚀 Analyze All Reviews", use_container_width=True)

                    if run_batch:
                        with st.spinner(f"Analyzing {len(batch_df):,} reviews..."):
                            texts = batch_df[review_col].astype(str).tolist()
                            cleaned_texts = [clean_review(t) for t in texts]
                            vectors = vectorizer.transform(cleaned_texts)
                            preds = model.predict(vectors)
                            probs = model.predict_proba(vectors).max(axis=1)

                            batch_df["predicted_sentiment"] = np.where(preds == 1, "Positive", "Negative")
                            batch_df["confidence"] = (probs * 100).round(1).astype(str) + "%"

                        st.success(f"Done — analyzed {len(batch_df):,} reviews.")

                        pos_count = int((preds == 1).sum())
                        neg_count = int((preds == 0).sum())
                        c1, c2 = st.columns(2)
                        c1.metric("😊 Positive reviews", f"{pos_count:,}")
                        c2.metric("😞 Negative reviews", f"{neg_count:,}")

                        fig_batch = go.Figure(go.Bar(
                            x=["Positive", "Negative"], y=[pos_count, neg_count],
                            marker_color=["#4ade80", "#f87171"],
                        ))
                        fig_batch.update_layout(
                            height=260, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#241f47"),
                        )
                        st.plotly_chart(fig_batch, use_container_width=True)

                        st.dataframe(batch_df, use_container_width=True)

                        csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download results as CSV",
                            data=csv_bytes,
                            file_name="intellireview_batch_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

# ---------------- TAB: History ----------------
with tab_history:
    with st.container(border=True):
        st.markdown("#### 📜 Your Analysis History")
        st.write("Every review you analyze in the **Analyze a Review** tab during this session is logged here.")

        if not st.session_state.history:
            st.info("No reviews analyzed yet in this session. Head to the **Analyze a Review** tab to get started.")
        else:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                csv_bytes = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download history as CSV",
                    data=csv_bytes,
                    file_name="intellireview_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                if st.button("🗑️ Clear history", use_container_width=True):
                    st.session_state.history = []
                    st.rerun()

# ---------------- TAB: Word Clouds ----------------
with tab_clouds:
    with st.container(border=True):
        st.markdown("#### 🔍 What words show up most in positive vs. negative reviews?")
        st.write(
            "Generated from all 50,000 reviews in the training dataset — bigger words "
            "appear more frequently in that class of review."
        )

        col_pos, col_neg = st.columns(2)
        with col_pos:
            st.markdown("**😊 Positive reviews**")
            try:
                st.image("positive_wordcloud.png", use_container_width=True)
            except Exception:
                st.info("Word cloud image not found. Run `train_model.py` to generate it.")
        with col_neg:
            st.markdown("**😞 Negative reviews**")
            try:
                st.image("negative_wordcloud.png", use_container_width=True)
            except Exception:
                st.info("Word cloud image not found. Run `train_model.py` to generate it.")

# ---------------- TAB 3: About ----------------
with tab3:
    with st.container(border=True):
        st.markdown("""
#### ℹ️ About this project

**IntelliReview** is a Natural Language Processing project that classifies movie reviews as
**positive** or **negative**, without ever being explicitly told the meaning of any word.

**Pipeline**
1. **Dataset** — the full 50,000-review IMDB Movie Reviews dataset (25,000 positive / 25,000 negative),
   a well-known benchmark dataset for binary sentiment classification.
2. **Cleaning** — HTML tags are stripped, contractions are expanded, and negation words
   (not/no/never/etc.) are preserved and used to tag the words they negate, so "not good"
   and "good" are treated as different signals rather than the negation being discarded.
3. **Train/test split** — 75% training, 25% testing, stratified to preserve class balance.
4. **Feature extraction** — TF-IDF (Term Frequency – Inverse Document Frequency) converts each
   review into a numeric vector, capturing which words matter most in each review relative to
   the whole dataset. Unigrams and bigrams are used, with a 30,000-word vocabulary.
5. **Model** — Logistic Regression, a simple and fast linear classifier that is a strong
   baseline for text classification tasks.
6. **Evaluation** — accuracy, F1-score, and a confusion matrix on the held-out test set.

**Beyond a basic classifier, this project also includes:**
- **Explainability** — every prediction shows the specific words that pushed it toward Positive or Negative, based on the model's own learned weights.
- **Batch analysis** — upload a CSV of reviews and get predictions for all of them at once.
- **Session history** — every review analyzed is logged and downloadable.
- **Word clouds** — the most common words in positive vs. negative reviews across the full dataset.

**Tech stack:** Python · Pandas · Scikit-learn · TF-IDF · Logistic Regression · Streamlit · Plotly · WordCloud

**A note on limitations:** like any bag-of-words model, this classifier can still be fooled by
sarcasm, idioms ("not bad" meaning "decent"), and mixed-sentiment reviews. A high confidence
score reflects how strongly a pattern matched the training data — not certainty about the truth.
        """)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("""
<div class="footer-credit">
    🎬 IntelliReview — AI Movie Review Sentiment Analyzer<br>
    Developed by <b>Pawan Dubey</b> · Built with Python, Scikit-learn &amp; Streamlit
</div>
""", unsafe_allow_html=True)