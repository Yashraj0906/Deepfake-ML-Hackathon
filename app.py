import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DetGuard — Deepfake Detector",
    page_icon="🛡️",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}
.main { background-color: #0d0f14; }
.stApp { background-color: #0d0f14; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.header-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 32px;
    margin-bottom: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.header-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 8px 0;
    letter-spacing: -1px;
}
.header-sub {
    font-size: 0.95rem;
    color: #718096;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.result-card {
    background: #1a1f2e;
    border-radius: 12px;
    padding: 28px;
    margin: 20px 0;
    border: 1px solid #2d3748;
    position: relative;
    overflow: hidden;
}
.result-real { border-left: 4px solid #2ecc71; }
.result-fake { border-left: 4px solid #e74c3c; }
.verdict-text {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -2px;
}
.verdict-real { color: #2ecc71; }
.verdict-fake { color: #e74c3c; }
.confidence-text { font-size: 1.1rem; color: #a0aec0; margin: 4px 0 0 0; }
.metric-box {
    background: #0d1117;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #2d3748;
}
.metric-label {
    font-size: 0.75rem;
    color: #718096;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
    margin-top: 4px;
}
.risk-badge {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 12px;
    font-family: 'Space Mono', monospace;
}
.risk-low      { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; }
.risk-medium   { background: #3a3010; color: #f1c40f; border: 1px solid #f1c40f; }
.risk-high     { background: #3a2010; color: #e67e22; border: 1px solid #e67e22; }
.risk-critical { background: #3a1010; color: #e74c3c; border: 1px solid #e74c3c; }
.bar-container {
    background: #0d1117;
    border-radius: 6px;
    height: 10px;
    margin-top: 6px;
    overflow: hidden;
}
.bar-fill-real { background: linear-gradient(90deg, #27ae60, #2ecc71); height: 100%; border-radius: 6px; }
.bar-fill-fake { background: linear-gradient(90deg, #c0392b, #e74c3c); height: 100%; border-radius: 6px; }
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #1a1f2e;
}
.model-name { font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #a0aec0; }
.model-score { font-family: 'Space Mono', monospace; font-size: 0.9rem; font-weight: 700; }
.cam-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    margin-bottom: 6px;
    text-align: center;
}
.footer {
    text-align: center;
    color: #4a5568;
    font-size: 0.8rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #1a1f2e;
    font-family: 'Space Mono', monospace;
}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        effnet = tf.keras.models.load_model('best_effnet_model.keras')
        resnet = tf.keras.models.load_model('best_resnet_model.keras')
        return effnet, resnet
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")
        st.info("Make sure best_effnet_model.keras and best_resnet_model.keras are in the same folder as app.py")
        return None, None

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pil_image, effnet_model, resnet_model, w_eff=0.43, w_res=0.57, img_size=224):
    img    = pil_image.convert('RGB').resize((img_size, img_size))
    arr    = np.array(img, dtype='float32')[np.newaxis]
    eff_in = tf.keras.applications.efficientnet_v2.preprocess_input(arr.copy())
    res_in = tf.keras.applications.resnet_v2.preprocess_input(arr.copy())
    p_eff  = float(effnet_model.predict(eff_in, verbose=0)[0][0])
    p_res  = float(resnet_model.predict(res_in, verbose=0)[0][0])
    prob   = p_eff * w_eff + p_res * w_res
    label  = 'REAL' if prob >= 0.5 else 'FAKE'
    conf   = prob if prob >= 0.5 else 1 - prob

    if   label == 'REAL' and conf >= 0.90: risk = ('🟢 LOW RISK — Authentic',          'risk-low')
    elif label == 'REAL' and conf >= 0.75: risk = ('🟡 MEDIUM RISK — Review',           'risk-medium')
    elif label == 'REAL' and conf <  0.75: risk = ('🟠 LOW CONFIDENCE — Manual Check',  'risk-high')
    elif label == 'FAKE' and conf <  0.80: risk = ('🟠 HIGH RISK — Possible Deepfake',  'risk-high')
    else:                                   risk = ('🔴 CRITICAL — Deepfake Detected',   'risk-critical')

    return {
        'label'     : label,
        'confidence': round(conf * 100, 1),
        'prob_real' : round(prob * 100, 1),
        'prob_fake' : round((1 - prob) * 100, 1),
        'p_effnet'  : round(p_eff * 100, 1),
        'p_resnet'  : round(p_res * 100, 1),
        'risk_text' : risk[0],
        'risk_class': risk[1],
        'eff_in'    : eff_in,
    }

# ── UI ────────────────────────────────────────────────────────────────────────

# Header — FaceGuard
st.markdown("""
<div class="header-box">
    <p class="header-title">🛡️ DetGuard</p>
    <p class="header-sub">AI-Powered Deepfake Detection &nbsp;·&nbsp; EfficientNetV2B2 + ResNet101V2 Ensemble &nbsp;·&nbsp; 90.33% Accuracy</p>
</div>
""", unsafe_allow_html=True)

effnet_model, resnet_model = load_models()
if effnet_model is None:
    st.stop()

st.markdown("### Upload an Image")
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload",
    type=['jpg', 'jpeg', 'png'],
    label_visibility='collapsed'
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.markdown("**Uploaded Image**")
        st.image(image, use_container_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            result = predict(image, effnet_model, resnet_model)

        label      = result['label']
        card_class = 'result-real' if label == 'REAL' else 'result-fake'
        text_class = 'verdict-real' if label == 'REAL' else 'verdict-fake'

        st.markdown(f"""
        <div class="result-card {card_class}">
            <p class="verdict-text {text_class}">{label}</p>
            <p class="confidence-text">{result['confidence']}% confidence</p>
            <span class="risk-badge {result['risk_class']}">{result['risk_text']}</span>
        </div>
        """, unsafe_allow_html=True)

    # Scores
    st.markdown("---")
    st.markdown("### Detection Scores")
    col_r, col_f = st.columns(2)
    with col_r:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Real Probability</div>
            <div class="metric-value" style="color:#2ecc71">{result['prob_real']}%</div>
            <div class="bar-container"><div class="bar-fill-real" style="width:{result['prob_real']}%"></div></div>
        </div>""", unsafe_allow_html=True)
    with col_f:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Fake Probability</div>
            <div class="metric-value" style="color:#e74c3c">{result['prob_fake']}%</div>
            <div class="bar-container"><div class="bar-fill-fake" style="width:{result['prob_fake']}%"></div></div>
        </div>""", unsafe_allow_html=True)

    # Model breakdown
    st.markdown("### Model Breakdown")
    st.markdown(f"""
    <div class="result-card">
        <div class="model-row">
            <span class="model-name">EfficientNetV2B2 &nbsp;<span style="color:#4a5568">(weight: 43%)</span></span>
            <span class="model-score" style="color:{'#2ecc71' if result['p_effnet'] >= 50 else '#e74c3c'}">
                {'REAL' if result['p_effnet'] >= 50 else 'FAKE'} &nbsp; {result['p_effnet']}%
            </span>
        </div>
        <div class="model-row" style="border-bottom:none">
            <span class="model-name">ResNet101V2 &nbsp;<span style="color:#4a5568">(weight: 57%)</span></span>
            <span class="model-score" style="color:{'#2ecc71' if result['p_resnet'] >= 50 else '#e74c3c'}">
                {'REAL' if result['p_resnet'] >= 50 else 'FAKE'} &nbsp; {result['p_resnet']}%
            </span>
        </div>
        <div style="margin-top:14px; padding-top:14px; border-top:1px solid #2d3748;">
            <span class="model-name">ENSEMBLE RESULT &nbsp;<span style="color:#4a5568">(43% EfficientNet + 57% ResNet)</span></span>
            <span class="model-score" style="float:right; color:{'#2ecc71' if label == 'REAL' else '#e74c3c'}">
                {label} &nbsp; {result['confidence']}%
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; padding:48px; color:#4a5568;">
        <div style="font-size:3rem; margin-bottom:12px;">🔍</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.9rem;">
            Upload an image to begin detection
        </div>
        <div style="font-size:0.8rem; margin-top:8px; color:#2d3748;">
            Supports JPG, JPEG, PNG
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("ℹ️ About DetGuard"):
    st.markdown("""
    **DetGuard** is an AI-powered deepfake detection system using an ensemble of two deep learning models:

    - **EfficientNetV2B2** — detects subtle AI-generated texture artifacts
    - **ResNet101V2** — captures structural facial inconsistencies
    - **Ensemble** — weighted combination (43% + 57%) optimized on validation set
    - **Grad-CAM** — visualizes which image regions triggered the fake/real decision

    **Performance Metrics:**
    Accuracy: **90.33%** | AUC-ROC: **0.9475** | F1: **0.90** | Recall: **91.78%**
    """)

st.markdown("""
<div class="footer">
    🛡️ DetGuard &nbsp;·&nbsp; EfficientNetV2B2 + ResNet101V2 &nbsp;·&nbsp; Built by Yashraj Kumar
</div>
""", unsafe_allow_html=True)