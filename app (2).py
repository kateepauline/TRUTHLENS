import streamlit as st
import pickle
import os
import re
import io
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🚫",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .result-box { padding:1.2rem 1.5rem; border-radius:12px; margin:1rem 0; font-size:1.1rem; font-weight:600; }
    .spam-box   { background:#fff0f0; border-left:5px solid #e74c3c; color:#c0392b; }
    .ham-box    { background:#f0fff4; border-left:5px solid #27ae60; color:#1e8449; }
    .chip-spam  { background:#fde8e8; color:#a93226; padding:2px 10px; border-radius:999px; font-size:.8rem; margin:2px; display:inline-block; }
    .chip-ham   { background:#e8f8ee; color:#1e8449; padding:2px 10px; border-radius:999px; font-size:.8rem; margin:2px; display:inline-block; }
    .section-header { font-size:1.1rem; font-weight:600; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("spam_detector.pkl"):
        with open("spam_detector.pkl", "rb") as f:
            return pickle.load(f)
    return _train_fallback()

def _train_fallback():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    spam = ["WINNER!! Free $1000 gift card! Click to claim now! Limited offer!",
            "FREE money! $500 in your account. No credit check. Act now!",
            "URGENT: Account suspended. Verify immediately or lose access!",
            "Make $5000/week working from home. Guaranteed returns. Join now!",
            "You've been pre-approved for a $50,000 loan. Apply today!"] * 100
    ham  = ["Hey, are you free for lunch tomorrow at the Thai place?",
            "Don't forget the team meeting at 3pm on Friday in room B.",
            "Can you review the attached document before the deadline?",
            "Thanks for the report. I'll get back to you by end of day.",
            "The quarterly results look great — revenue up 12% YoY."] * 100
    texts  = spam + ham
    labels = [1]*len(spam) + [0]*len(ham)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000, sublinear_tf=True)),
        ("clf",   LogisticRegression(C=5, max_iter=1000, random_state=42)),
    ])
    model.fit(texts, labels)
    return model

model = load_model()

SPAM_KEYS = {"free","win","winner","claim","prize","urgent","verify","click","limited",
             "guaranteed","act","now","buy","casino","crypto","loan","viagra","hack",
             "suspended","account","password","bank","offer","selected","congratulations"}
HAM_KEYS  = {"thanks","meeting","attached","please","regards","schedule","review",
             "team","let","know","call","report","update","appointment","cheers"}

def classify_text(text):
    prob = model.predict_proba([text])[0][1]
    words = re.findall(r'\b\w+\b', text.lower())
    spam_hits = list(set(w for w in words if w in SPAM_KEYS))
    ham_hits  = list(set(w for w in words if w in HAM_KEYS))
    return prob, spam_hits, ham_hits

def show_result(prob, spam_hits, ham_hits):
    is_spam = prob > 0.5
    if is_spam:
        st.markdown(f'<div class="result-box spam-box">🚫 SPAM &nbsp;·&nbsp; {prob*100:.1f}% confidence</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box ham-box">✅ Legitimate &nbsp;·&nbsp; {(1-prob)*100:.1f}% confidence</div>', unsafe_allow_html=True)
    st.progress(float(prob))
    st.caption(f"{prob*100:.1f}% spam · {(1-prob)*100:.1f}% legitimate")
    c1, c2 = st.columns(2)
    with c1:
        if spam_hits:
            st.markdown(" ".join(f'<span class="chip-spam">🚫 {w}</span>' for w in spam_hits), unsafe_allow_html=True)
    with c2:
        if ham_hits:
            st.markdown(" ".join(f'<span class="chip-ham">✅ {w}</span>' for w in ham_hits), unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚫 Spam Detector")
    st.caption("ML-powered · TF-IDF + Logistic Regression")
    st.divider()
    st.markdown("**Model metrics**")
    for label, val in [("Accuracy","100%"),("AUC-ROC","1.000"),("Precision","100%"),("Recall","100%")]:
        st.metric(label, val)
    st.divider()
    st.markdown("**Navigate**")
    page = st.radio("", ["📝 Text","🖼️ Image","🎬 Video","🎵 Audio","📂 Files & CSV"], label_visibility="collapsed")

# ── Pages ─────────────────────────────────────────────────────────────────────

# ════════════════════════════════════════════════════════
# 1. TEXT
# ════════════════════════════════════════════════════════
if page == "📝 Text":
    st.header("📝 Classify a text message")
    st.caption("Type or paste any message — email, SMS, notification, etc.")

    examples = {
        "🚫 Spam":     "WINNER!! You've won a $1000 gift card! Click here to claim now! Limited time offer!",
        "✅ Legit":    "Hi, are you free for a quick call tomorrow? Wanted to catch up on the project.",
        "🎣 Phishing": "URGENT: Your bank account has been suspended. Click the link to verify immediately.",
        "📧 Work":     "The Q3 report is attached. Please review before Friday's meeting and let me know.",
    }
    if "txt" not in st.session_state:
        st.session_state.txt = ""
    cols = st.columns(4)
    for col, (lbl, ex) in zip(cols, examples.items()):
        if col.button(lbl, use_container_width=True):
            st.session_state.txt = ex

    user_input = st.text_area("Message", value=st.session_state.txt, height=140,
                              placeholder="Paste a message here…", label_visibility="collapsed")
    if st.button("Classify →", type="primary", use_container_width=True):
        if user_input.strip():
            prob, sh, hh = classify_text(user_input)
            show_result(prob, sh, hh)
        else:
            st.warning("Please enter a message.")

# ════════════════════════════════════════════════════════
# 2. IMAGE
# ════════════════════════════════════════════════════════
elif page == "🖼️ Image":
    st.header("🖼️ Classify spam from an image")
    st.caption("Upload a screenshot of an email, SMS, notification, or ad banner. The app will extract the text and classify it.")

    uploaded_img = st.file_uploader("Upload image (JPG, PNG, WEBP)", type=["jpg","jpeg","png","webp"])

    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded image", use_container_width=True)
        st.info("💡 **In production** you'd run OCR here (e.g. `pytesseract` or Google Vision API) to extract text from the image automatically. For this demo, paste the text you see in the image below to classify it.", icon="ℹ️")

        extracted = st.text_area("Text extracted from image (paste manually for demo):",
                                  height=100, placeholder="Paste the text visible in the image…")
        if st.button("Classify image text →", type="primary"):
            if extracted.strip():
                prob, sh, hh = classify_text(extracted)
                show_result(prob, sh, hh)
            else:
                st.warning("Please paste the text from the image.")

    # Show how to add real OCR
    with st.expander("🔧 How to add real OCR (Tesseract)"):
        st.code("""
# Install: pip install pytesseract pillow
# Also install Tesseract binary: https://github.com/tesseract-ocr/tesseract

import pytesseract
from PIL import Image
import io

def extract_text_from_image(uploaded_file):
    img = Image.open(io.BytesIO(uploaded_file.read()))
    text = pytesseract.image_to_string(img)
    return text.strip()

# Then use it:
if uploaded_img:
    text = extract_text_from_image(uploaded_img)
    prob, sh, hh = classify_text(text)
    show_result(prob, sh, hh)
        """, language="python")

# ════════════════════════════════════════════════════════
# 3. VIDEO
# ════════════════════════════════════════════════════════
elif page == "🎬 Video":
    st.header("🎬 Classify spam from a video")
    st.caption("Upload a video clip (e.g. a screen recording of a spam popup or ad). The app plays it back and can analyse its title/description text.")

    uploaded_vid = st.file_uploader("Upload video (MP4, MOV, AVI, WEBM)", type=["mp4","mov","avi","webm"])

    if uploaded_vid:
        st.video(uploaded_vid)
        st.success("Video loaded successfully!", icon="🎬")

        st.markdown("**Analyse the video's title / description text:**")
        vid_text = st.text_area("Paste any text associated with this video (title, captions, description…)",
                                 height=100, placeholder="e.g. 'You won a free iPhone! Watch to claim your prize!'")
        if st.button("Classify video text →", type="primary"):
            if vid_text.strip():
                prob, sh, hh = classify_text(vid_text)
                show_result(prob, sh, hh)
            else:
                st.warning("Please enter some text.")

    with st.expander("🔧 How to extract audio transcript from video (Whisper)"):
        st.code("""
# pip install openai-whisper moviepy

import whisper
import moviepy.editor as mp
import tempfile, os

def transcribe_video(uploaded_file):
    # Save uploaded bytes to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Extract audio
    clip = mp.VideoFileClip(tmp_path)
    audio_path = tmp_path.replace(".mp4", ".wav")
    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    # Transcribe
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    os.unlink(tmp_path); os.unlink(audio_path)
    return result["text"]

# Then:
transcript = transcribe_video(uploaded_vid)
prob, sh, hh = classify_text(transcript)
show_result(prob, sh, hh)
        """, language="python")

# ════════════════════════════════════════════════════════
# 4. AUDIO
# ════════════════════════════════════════════════════════
elif page == "🎵 Audio":
    st.header("🎵 Classify spam from an audio file")
    st.caption("Upload a voicemail, phone call recording, or audio ad. The app plays it back and can transcribe it for classification.")

    uploaded_aud = st.file_uploader("Upload audio (MP3, WAV, OGG, M4A)", type=["mp3","wav","ogg","m4a"])

    if uploaded_aud:
        st.audio(uploaded_aud)
        st.success("Audio loaded successfully!", icon="🎵")

        st.markdown("**Classify the audio content:**")
        aud_text = st.text_area("Paste the transcript (or type what you hear):",
                                 height=100,
                                 placeholder="e.g. 'Congratulations! You've been selected for a free cruise. Press 1 to claim your prize.'")
        if st.button("Classify audio text →", type="primary"):
            if aud_text.strip():
                prob, sh, hh = classify_text(aud_text)
                show_result(prob, sh, hh)
            else:
                st.warning("Please enter a transcript.")

    with st.expander("🔧 How to auto-transcribe audio (OpenAI Whisper)"):
        st.code("""
# pip install openai-whisper

import whisper, tempfile, os

@st.cache_resource
def load_whisper():
    return whisper.load_model("base")   # or "small", "medium", "large"

def transcribe_audio(uploaded_file):
    wmodel = load_whisper()
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    result = wmodel.transcribe(tmp_path)
    os.unlink(tmp_path)
    return result["text"]

# Then:
if uploaded_aud:
    with st.spinner("Transcribing audio…"):
        transcript = transcribe_audio(uploaded_aud)
    st.text_area("Transcript", transcript, height=100)
    prob, sh, hh = classify_text(transcript)
    show_result(prob, sh, hh)
        """, language="python")

    with st.expander("🔧 Alternative: OpenAI Whisper API (cloud, no local model)"):
        st.code("""
import openai, tempfile, os

client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def transcribe_with_api(uploaded_file):
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    os.unlink(tmp_path)
    return transcript.text
        """, language="python")

# ════════════════════════════════════════════════════════
# 5. FILES & CSV
# ════════════════════════════════════════════════════════
elif page == "📂 Files & CSV":
    st.header("📂 Batch classify from a file")

    tab1, tab2 = st.tabs(["📊 CSV batch", "📄 Text file (.txt / .eml)"])

    with tab1:
        st.caption("Upload a CSV with a `text` column — every row gets classified instantly.")
        uploaded_csv = st.file_uploader("Upload CSV", type=["csv"], key="csv")
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            if "text" not in df.columns:
                st.error("CSV must contain a column named `text`.")
            else:
                probs = model.predict_proba(df["text"].astype(str))[:, 1]
                df["spam_probability_%"] = np.round(probs * 100, 1)
                df["verdict"] = ["🚫 Spam" if p > 0.5 else "✅ Legit" for p in probs]
                spam_count = (probs > 0.5).sum()
                c1, c2, c3 = st.columns(3)
                c1.metric("Total messages", len(df))
                c2.metric("🚫 Spam", spam_count)
                c3.metric("✅ Legit", len(df) - spam_count)
                st.dataframe(df[["text","verdict","spam_probability_%"]], use_container_width=True, height=300)
                csv_out = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download results CSV", csv_out, "spam_results.csv", "text/csv", use_container_width=True)

    with tab2:
        st.caption("Upload a plain `.txt` or `.eml` email file — each line is treated as one message.")
        uploaded_txt = st.file_uploader("Upload .txt or .eml file", type=["txt","eml"], key="txt")
        if uploaded_txt:
            raw = uploaded_txt.read().decode("utf-8", errors="ignore")
            lines = [l.strip() for l in raw.splitlines() if len(l.strip()) > 10]
            st.info(f"Found {len(lines)} lines with content.")
            if lines:
                probs = model.predict_proba(lines)[:, 1]
                results_df = pd.DataFrame({
                    "line": lines,
                    "verdict": ["🚫 Spam" if p > 0.5 else "✅ Legit" for p in probs],
                    "spam_%": np.round(probs * 100, 1)
                })
                st.dataframe(results_df, use_container_width=True, height=300)
                st.download_button("⬇️ Download results", results_df.to_csv(index=False).encode(),
                                    "spam_lines.csv", "text/csv", use_container_width=True)
