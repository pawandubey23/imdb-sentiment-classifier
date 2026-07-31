"""
🎬 CineSense — AI Movie Review Sentiment Analyzer
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
    page_title="CineSense | AI Sentiment Analyzer",
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
        background: radial-gradient(circle at 20% 0%, #1b1033 0%, #0d0a1f 45%, #060512 100%);
    }

    #MainMenu, footer, header {visibility: hidden;}

    .hero {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
    }
    .hero h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff6ec7, #7873f5, #4ade80);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #b9b3d9;
        font-size: 1.02rem;
        max-width: 640px;
        margin: 0 auto;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.5rem 1.6rem;
        backdrop-filter: blur(6px);
        margin-bottom: 1.1rem;
    }

    .metric-box {
        text-align: center;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.9rem 0.4rem;
    }
    .metric-box .val { font-size: 1.5rem; font-weight: 700; color: #fff; }
    .metric-box .lbl { font-size: 0.78rem; color: #9d97c2; text-transform: uppercase; letter-spacing: .04em;}

    .result-positive {
        background: linear-gradient(135deg, rgba(74,222,128,0.18), rgba(34,197,94,0.05));
        border: 1px solid rgba(74,222,128,0.4);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(248,113,113,0.18), rgba(239,68,68,0.05));
        border: 1px solid rgba(248,113,113,0.4);
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
    }
    .result-emoji { font-size: 3.2rem; }
    .result-label { font-size: 1.6rem; font-weight: 700; color: #fff; margin-top: 0.2rem;}
    .result-conf { color: #cfc9ec; font-size: 0.95rem; margin-top: 0.3rem;}

    .stTextArea textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 12px !important;
        color: #fff !important;
        font-size: 1rem !important;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #7873f5, #ff6ec7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover { transform: translateY(-2px); }

    .chip {
        display: inline-block;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        color: #d9d5f0;
        border-radius: 999px;
        padding: 0.35rem 0.85rem;
        margin: 0.2rem;
        font-size: 0.85rem;
        cursor: pointer;
    }

    .footer-credit {
        text-align: center;
        color: #85809f;
        font-size: 0.85rem;
        padding: 1.6rem 0 0.6rem 0;
        border-top: 1px solid rgba(255,255,255,0.07);
        margin-top: 2rem;
    }
    .footer-credit b { color: #c9c3ec; }
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


try:
    vectorizer, model, metrics = load_artifacts()
    artifacts_ready = True
except FileNotFoundError:
    artifacts_ready = False

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 CineSense")
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
    <h1>🎬 CineSense</h1>
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
tab1, tab2, tab3 = st.tabs(["🔮 Analyze a Review", "📊 Model Performance", "ℹ️ About the Project"])

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
                    font=dict(color="#e5e2f5"),
                    xaxis=dict(range=[0, 100], showgrid=False, title="Probability (%)"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, use_container_width=True)

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
            font=dict(color="#e5e2f5"),
            height=420,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

# ---------------- TAB 3: About ----------------
with tab3:
    with st.container(border=True):
        st.markdown("""
#### ℹ️ About this project

**CineSense** is a Natural Language Processing project that classifies movie reviews as
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

**Tech stack:** Python · Pandas · Scikit-learn · TF-IDF · Logistic Regression · Streamlit · Plotly

**A note on limitations:** like any bag-of-words model, this classifier can still be fooled by
sarcasm, idioms ("not bad" meaning "decent"), and mixed-sentiment reviews. A high confidence
score reflects how strongly a pattern matched the training data — not certainty about the truth.
        """)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown("""
<div class="footer-credit">
    🎬 CineSense — AI Movie Review Sentiment Analyzer<br>
    Developed by <b>Pawan Dubey</b> · Built with Python, Scikit-learn &amp; Streamlit
</div>
""", unsafe_allow_html=True)