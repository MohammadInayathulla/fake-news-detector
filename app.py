import streamlit as st
from groq import Groq
from lime.lime_text import LimeTextExplainer
import plotly.graph_objects as go
import joblib
import numpy as np
from PIL import Image
import re
import os
import pandas as pd
import plotly.express as px

# --- Optional Torch ---
use_torch = False
try:
    import torch
    import torchvision.transforms as T
    from torchvision import models
    use_torch = True
except:
    use_torch = False

# --- Load model ---
model = joblib.load("model_logreg.pkl")

# --- Sentence Transformer ---
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('all-MiniLM-L6-v2')
TEXT_DIM = 384

# --- Groq ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# --- Image setup ---
if use_torch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mobilenet_v2 = models.mobilenet_v2(pretrained=True).to(device)
    mobilenet_v2.eval()
    IMG_SIZE = (224, 224)
    transform_img = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    sample = Image.new("RGB", IMG_SIZE, (128, 128, 128))
    with torch.no_grad():
        x = transform_img(sample).unsqueeze(0).to(device)
        feats = mobilenet_v2.features(x)
        feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).reshape(1, -1)
        IMG_FEATURE_DIM = feats.shape[1]
else:
    IMG_FEATURE_DIM = 1280

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def clean_text_simple(text):
    if text is None:
        return ""
    text = re.sub(r"http\S+", " ", str(text))
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def extract_image_features_pil(pil_img):
    if not use_torch:
        return np.zeros((IMG_FEATURE_DIM,), dtype=float)
    try:
        x = transform_img(pil_img.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = mobilenet_v2.features(x)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).reshape(1, -1)
            return feats.cpu().numpy().flatten()
    except:
        return np.zeros((IMG_FEATURE_DIM,), dtype=float)

def dynamic_explanation(label, prob):
    if label == "FAKE":
        if prob > 0.85:
            return [
                "⚠️ Strong linguistic signs of misinformation.",
                "🧩 Image + text alignment appears suspicious.",
                "📉 Model highly confident this headline is fabricated."
            ]
        if prob > 0.60:
            return [
                "⚠️ Some patterns usually linked to misleading headlines.",
                "🔍 Consider verifying from a trusted source."
            ]
        return [
            "⚠️ Model slightly leans towards fake.",
            "📌 Double-check with external fact-checkers."
        ]
    else:
        if prob < 0.15:
            return [
                "🟢 Strong linguistic consistency detected.",
                "🌐 Headline structure matches verified news patterns.",
                "🧠 No major anomaly detected."
            ]
        if prob < 0.35:
            return [
                "🟢 Mostly consistent with real headlines.",
                "🔍 No major misleading cues detected."
            ]
        return [
            "🟢 Model slightly leans towards real.",
            "📌 Structure resembles real news writing."
        ]

def get_groq_explanation(headline, prediction, confidence, lime_words, has_image):
    fake_words = [w for w, v in lime_words if v > 0]
    real_words = [w for w, v in lime_words if v < 0]
    prompt = f"""You are an AI assistant helping explain a fake news detection result.

A multimodal AI system (using both text and image analysis) analyzed this news headline:
"{headline}"

Results:
- Prediction: {prediction}
- Confidence: {confidence * 100:.1f}%
- Image provided: {"Yes" if has_image else "No"}
- Top words pushing toward FAKE: {fake_words if fake_words else "None"}
- Top words pushing toward REAL: {real_words if real_words else "None"}

Write a clear, concise 3-sentence explanation of:
1. What the prediction means
2. Which specific words or patterns drove this decision
3. What the user should do next (verify, trust, etc.)

Keep it simple, factual, and helpful. Do not use bullet points."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.4
    )
    return response.choices[0].message.content.strip()

# --- LIME setup ---
lime_explainer = LimeTextExplainer(class_names=["REAL", "FAKE"])

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown("""
<style>
.result-banner {
    padding: 15px;
    border-radius: 10px;
    color: white;
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 15px;
}
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background: #f7f7f7;
    text-align: center;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection — Hybrid AI System")

# --- Session state ---
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "word_weights" not in st.session_state:
    st.session_state.word_weights = []
if "label" not in st.session_state:
    st.session_state.label = None
if "prob" not in st.session_state:
    st.session_state.prob = None
if "cleaned_text" not in st.session_state:
    st.session_state.cleaned_text = None
if "has_image" not in st.session_state:
    st.session_state.has_image = False
if "img_vec_stored" not in st.session_state:
    st.session_state.img_vec_stored = None

st.sidebar.header("⚙️ Options")
expert_mode = st.sidebar.checkbox("Enable Expert Mode")
st.sidebar.info("Multimodal: Sentence Transformers + MobileNetV2")
st.sidebar.markdown("**Best Model:** Logistic Regression — 88.5%")

col1, col2 = st.columns([2, 1])
with col1:
    text = st.text_area("Enter Headline", height=140)
with col2:
    upload = st.file_uploader("Upload related image (optional)", type=["png", "jpg", "jpeg"])

# ----------------------------------------------------------------------
# ANALYZE BUTTON
# ----------------------------------------------------------------------
if st.button("🔍 Analyze", use_container_width=True):

    if not text.strip():
        st.warning("Please enter a headline.")
        st.stop()

    if len(text.strip()) < 20:
        st.error("Title too short or incorrect news title.")
        st.stop()

    cleaned = clean_text_simple(text)
    word_weights = []  # safe default in case LIME fails
    text_vec = st_model.encode([cleaned])

    if upload:
        pil_img = Image.open(upload)
        img_vec = extract_image_features_pil(pil_img).reshape(1, -1)
        has_image = True
    else:
        img_vec = np.zeros((1, IMG_FEATURE_DIM))
        has_image = False

    final_vec = np.concatenate([text_vec, img_vec], axis=1)

    expected = model.n_features_in_
    if final_vec.shape[1] != expected:
        st.error(f"Feature mismatch: {final_vec.shape[1]} vs expected {expected}")
        st.stop()

    prob = model.predict_proba(final_vec)[0][1]
    label = "FAKE" if prob >= 0.5 else "REAL"
    conf = round(prob * 100, 2)
    banner_color = "#e63946" if label == "FAKE" else "#2a9d8f"

    st.markdown(
        f"<div class='result-banner' style='background:{banner_color};'>"
        f"Prediction: {label} — Confidence: {conf}%"
        f"</div>",
        unsafe_allow_html=True
    )

    for line in dynamic_explanation(label, prob):
        st.write(line)

    if upload:
        st.image(pil_img, width=350)

    # --- LIME ---
    st.markdown("---")
    st.subheader("🔍 Why did the model decide this?")

    actual_img_vec = img_vec.copy()

    def predict_for_lime(texts_list):
        text_vecs = st_model.encode(texts_list, show_progress_bar=False)
        img_vecs = np.tile(actual_img_vec, (len(texts_list), 1))
        combined = np.concatenate([text_vecs, img_vecs], axis=1)
        return model.predict_proba(combined)

    try:
        with st.spinner("Generating word-level explanation (~15 seconds)..."):
            exp = lime_explainer.explain_instance(
                cleaned,
                predict_for_lime,
                num_features=10,
                num_samples=300
            )

        word_weights = exp.as_list()
        fake_words = [(w, v) for w, v in word_weights if v > 0]
        real_words = [(w, v) for w, v in word_weights if v < 0]

        col_fake, col_real = st.columns(2)

        with col_fake:
            st.markdown("##### 🔴 Words pushing toward FAKE")
            if fake_words:
                fig = go.Figure(go.Bar(
                    x=[v for _, v in fake_words],
                    y=[w for w, _ in fake_words],
                    orientation='h',
                    marker_color='#e63946'
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Weight")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strong FAKE word indicators found.")

        with col_real:
            st.markdown("##### 🟢 Words pushing toward REAL")
            if real_words:
                fig = go.Figure(go.Bar(
                    x=[abs(v) for _, v in real_words],
                    y=[w for w, _ in real_words],
                    orientation='h',
                    marker_color='#2a9d8f'
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10), xaxis_title="Weight")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strong REAL word indicators found.")

        st.markdown("##### 📝 Highlighted Headline")
        st.components.v1.html(exp.as_html(), height=200, scrolling=True)

    except Exception as e:
        st.warning(f"⚠️ LIME explanation unavailable: {e}")

    # --- Groq Explanation ---
    st.markdown("---")
    st.subheader("🤖 AI Explanation")

    with st.spinner("Generating AI explanation via Groq..."):
        try:
            groq_explanation = get_groq_explanation(
                headline=text,
                prediction=label,
                confidence=prob if prob >= 0.5 else 1 - prob,
                lime_words=word_weights,
                has_image=has_image
            )
            st.markdown(
                f"""
                <div style='
                    background: {banner_color}18;
                    border-left: 4px solid {banner_color};
                    padding: 15px;
                    border-radius: 8px;
                    font-size: 15px;
                    line-height: 1.6;
                '>
                    {groq_explanation}
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.warning("⚠️ AI Explanation unavailable. Check your GROQ_API_KEY secret in HF Settings.")
            st.caption(f"Error detail: {e}")

    # Save to session state for expert mode
    st.session_state.analysis_done = True
    st.session_state.word_weights = word_weights
    st.session_state.label = label
    st.session_state.prob = prob
    st.session_state.cleaned_text = cleaned
    st.session_state.has_image = has_image

# ----------------------------------------------------------------------
# EXPERT MODE
# ----------------------------------------------------------------------
if st.session_state.analysis_done and expert_mode:

    metrics = {
        "Logistic Regression (LR)": {
            "Accuracy": 0.8850, "Precision": 0.8851,
            "Recall": 0.8850, "F1": 0.8850
        },
        "Random Forest (RF)": {
            "Accuracy": 0.8258, "Precision": 0.8259,
            "Recall": 0.8258, "F1": 0.8258
        },
        "Gradient Boosting (GB)": {
            "Accuracy": 0.8458, "Precision": 0.8472,
            "Recall": 0.8458, "F1": 0.8457
        }
    }

    df = pd.DataFrame(metrics).T

    st.markdown("---")
    st.header("🧪 Expert Mode Insights")

    tabs = st.tabs(["📊 Metrics Table", "📈 Bar Chart", "🏆 Best Model"])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            "<div class='metric-card' style='border-left:6px solid #1f77b4;'>"
            "<h4>🔵 Logistic Regression</h4>"
            f"{df.loc['Logistic Regression (LR)'].to_frame().T.to_html(index=False)}"
            "</div>", unsafe_allow_html=True
        )
        c2.markdown(
            "<div class='metric-card' style='border-left:6px solid #17becf;'>"
            "<h4>🔷 Random Forest</h4>"
            f"{df.loc['Random Forest (RF)'].to_frame().T.to_html(index=False)}"
            "</div>", unsafe_allow_html=True
        )
        c3.markdown(
            "<div class='metric-card' style='border-left:6px solid #e63946;'>"
            "<h4>🔺 Gradient Boosting</h4>"
            f"{df.loc['Gradient Boosting (GB)'].to_frame().T.to_html(index=False)}"
            "</div>", unsafe_allow_html=True
        )

    with tabs[1]:
        st.write("### 📈 Metric Comparison Bar Chart")
        metric_choice = st.selectbox("Choose a metric:", df.columns)
        fig = px.bar(
            df, x=df.index, y=metric_choice, color=df.index,
            color_discrete_map={
                "Logistic Regression (LR)": "#1f77b4",
                "Random Forest (RF)": "#17becf",
                "Gradient Boosting (GB)": "#e63946"
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        best = max(metrics.items(), key=lambda x: x[1]["F1"])[0]
        st.success(f"🏆 **Best Model Based on F1-Score: {best}**")
        st.download_button(
            "⬇ Download Metrics CSV",
            df.to_csv().encode(),
            "model_metrics.csv",
            "text/csv"
        )
