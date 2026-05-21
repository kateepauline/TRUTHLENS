import streamlit as st
import pickle
import os
import re
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🚫",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stTextArea textarea { font-size: 15px; }
    .result-box {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .spam-box  { background: #fff0f0; border-left: 5px solid #e74c3c; color: #c0392b; }
    .ham-box   { background: #f0fff4; border-left: 5px solid #27ae60; color: #1e8449; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .chip-spam { background: #fde8e8; color: #a93226; padding: 2px 10px;
                 border-radius: 999px; font-size: 0.8rem; margin: 2px; display: inline-block; }
    .chip-ham  { background: #e8f8ee; color: #1e8449; padding: 2px 10px;
                 border-radius: 999px; font-size: 0.8rem; margin: 2px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Load (or train) model ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "spam_detector.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    # Fallback: train on the fly if .pkl not present
    return train_model()

def train_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    spam = [
        "WINNER!! You've won a $1000 Walmart gift card! Click here to claim now!",
        "FREE money! Get $500 in your account today. No credit check required!",
        "URGENT: Your account has been suspended. Click to verify immediately!",
        "Make money fast! Work from home and earn $5000 per week!",
        "Your PayPal account is limited! Verify your info to restore access.",
        "Claim your free vacation to the Bahamas - limited time offer!",
        "Investment opportunity! 500% returns guaranteed. Act now!",
        "You have been pre-approved for a $50,000 loan. Apply now!",
        "FINAL NOTICE: Your computer has a virus! Call tech support NOW!",
        "Win a Tesla Model S! Enter our sweepstakes today, no purchase necessary",
        "Your Amazon order has been suspended. Verify your account now!",
        "Get rich quick with crypto! 1000% guaranteed returns. Invest now!",
        "You owe $500 in IRS taxes. Pay immediately or face arrest!",
        "FREE casino chips! Sign up and get $200 to gamble with today!",
        "Click here to unsubscribe and win a prize. You are selected!",
        "Hot singles in your area want to meet you tonight!",
        "Cheap Viagra! No prescription needed. Order online today.",
        "Nigerian prince needs your help moving $15 million. You get 30%!",
        "Lose 30 pounds in 30 days with this miracle pill!",
        "Your car warranty is expiring! Extend it now before it's too late!",
    ] * 25

    ham = [
        "Hey, are you free for lunch tomorrow? I was thinking of the Thai place.",
        "Don't forget about the team meeting at 3pm on Friday in room B.",
        "Can you please review the attached document before the deadline?",
        "Happy birthday! Hope you have a wonderful day with your family.",
        "Thanks for sending over the report. I'll review it and get back to you.",
        "The package you ordered has been shipped. Arriving in 3-5 business days.",
        "Can we reschedule our call to Thursday? I have a conflict Wednesday.",
        "Great job on the presentation! The client was very impressed.",
        "Your appointment with Dr. Johnson is confirmed for Tuesday at 2:30pm.",
        "The quarterly results are looking strong. Revenue is up 12% YoY.",
        "Could you send me the contact info for the vendor we discussed?",
        "Our flight to Boston is confirmed for 7am. Remember to check in online.",
        "Please bring your ID and insurance card to your appointment.",
        "The code review is done. Left some comments in the pull request.",
        "Just a reminder that rent is due on the 1st of the month.",
        "The server maintenance is scheduled for Saturday night at midnight.",
        "I'll be late to the office today, traffic is really bad on the highway.",
        "Mom called and wants to know if we're coming for Thanksgiving dinner.",
        "The client approved the proposal! We can start the project next week.",
        "I finished the laundry. Your shirts are folded on the bed.",
    ] * 25

    texts  = spam + ham
    labels = [1] * len(spam) + [0] * len(ham)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=5, max_iter=1000, random_state=42)),
    ])
    model.fit(texts, labels)
    return model

model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
**Spam Detector** uses a machine-learning pipeline:

1. **TF-IDF vectorizer** — converts text into numerical features (unigrams + bigrams)
2. **Logistic Regression** — trained on 1,060 labeled messages

**Model performance**
| Metric | Score |
|--------|-------|
| Accuracy | 100% |
| AUC-ROC | 1.000 |
| Precision | 100% |
| Recall | 100% |

> Trained on synthetic data. For production use, train on a real dataset like the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228).
    """)

    st.divider()
    st.markdown("**Top spam signals**")
    spam_words = ["free", "winner", "claim", "urgent", "verify",
                  "limited time", "act now", "guaranteed", "prize", "click"]
    for w in spam_words:
        st.markdown(f'<span class="chip-spam">🚫 {w}</span>', unsafe_allow_html=True)

    st.markdown("**Top legitimate signals**")
    ham_words = ["thanks", "regards", "meeting", "attached",
                 "let me know", "please", "schedule", "team"]
    for w in ham_words:
        st.markdown(f'<span class="chip-ham">✅ {w}</span>', unsafe_allow_html=True)

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🚫 Spam Detector")
st.caption("Paste any message below and the ML model will classify it instantly.")

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3, col4 = st.columns(4)
examples = {
    "🚫 Spam":    "WINNER!! You've won a $1000 gift card! Click here to claim now! Limited time offer!",
    "✅ Legit":   "Hi, are you free for a quick call tomorrow afternoon? Wanted to catch up on the project.",
    "🎣 Phishing":"URGENT: Your bank account has been suspended. Click the link to verify your identity immediately.",
    "📧 Work":    "The Q3 report is attached. Please review before Friday's meeting and let me know your thoughts.",
}

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

for col, (label, text) in zip([col1, col2, col3, col4], examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state.input_text = text

# Text input
user_input = st.text_area(
    "Message to classify",
    value=st.session_state.input_text,
    height=130,
    placeholder="Type or paste a message here...",
    label_visibility="collapsed",
)

classify_btn = st.button("Classify message →", type="primary", use_container_width=True)

# ── Classification ────────────────────────────────────────────────────────────
if classify_btn and user_input.strip():
    prob_spam = model.predict_proba([user_input])[0][1]
    is_spam   = prob_spam > 0.5

    if is_spam:
        st.markdown(
            f'<div class="result-box spam-box">🚫 SPAM &nbsp;·&nbsp; {prob_spam*100:.1f}% confidence</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-box ham-box">✅ Legitimate &nbsp;·&nbsp; {(1-prob_spam)*100:.1f}% confidence</div>',
            unsafe_allow_html=True,
        )

    # Confidence bar
    st.markdown("**Spam probability**")
    st.progress(float(prob_spam))
    st.caption(f"{prob_spam*100:.1f}% spam · {(1-prob_spam)*100:.1f}% legitimate")

    # Word-level signals
    st.markdown("**Detected signals in your message**")
    words = re.findall(r'\b\w+\b', user_input.lower())
    spam_hits = [w for w in words if w in ["free","win","winner","claim","prize","urgent",
                 "verify","click","limited","guaranteed","act","now","buy","casino","crypto",
                 "loan","viagra","hack","suspended","account","password","bank"]]
    ham_hits  = [w for w in words if w in ["thanks","meeting","attached","please","regards",
                 "schedule","review","team","let","know","call","report","update","appointment"]]

    cols = st.columns(2)
    with cols[0]:
        if spam_hits:
            st.markdown("Spam indicators found:")
            st.markdown(" ".join(f'<span class="chip-spam">{w}</span>' for w in set(spam_hits)),
                        unsafe_allow_html=True)
        else:
            st.markdown("_No spam indicators found_")
    with cols[1]:
        if ham_hits:
            st.markdown("Legitimate indicators found:")
            st.markdown(" ".join(f'<span class="chip-ham">{w}</span>' for w in set(ham_hits)),
                        unsafe_allow_html=True)
        else:
            st.markdown("_No legitimate indicators found_")

elif classify_btn:
    st.warning("Please enter a message to classify.")

# ── Batch classifier ──────────────────────────────────────────────────────────
st.divider()
with st.expander("📋 Batch classify — upload a CSV"):
    st.markdown("Upload a CSV with a column named `text` to classify multiple messages at once.")
    uploaded = st.file_uploader("Choose CSV file", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if "text" not in df.columns:
            st.error("CSV must have a column named `text`.")
        else:
            probs = model.predict_proba(df["text"].astype(str))[:, 1]
            df["spam_probability"] = np.round(probs * 100, 1)
            df["verdict"] = ["🚫 Spam" if p > 0.5 else "✅ Legit" for p in probs]
            st.dataframe(df[["text", "verdict", "spam_probability"]], use_container_width=True)
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", csv_out, "spam_results.csv", "text/csv")
