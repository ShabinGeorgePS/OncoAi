
import streamlit as st
from PIL import Image
import sys, os
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from predictor import predict, get_model
from gradcam import generate_gradcam

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ONCOAi — Oral Cancer Detection",
    page_icon="🦷",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: #0f1117;
}
[data-testid="stSidebar"] {
    background: #161b27;
    border-right: 1px solid #2d3748;
}

/* ── Header ── */
.onco-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.5rem 0 0.5rem 0;
}
.onco-title {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00c9ff, #028090);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.onco-sub {
    font-size: 0.95rem;
    color: #94a3b8;
    margin: 0;
    letter-spacing: 0.05em;
}
.onco-badge {
    background: #028090;
    color: white;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.8rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* ── Result cards ── */
.cancer-box {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #2d1515, #3d1f1f);
    border: 1px solid #e74c3c;
    border-left: 5px solid #e74c3c;
}
.safe-box {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    background: linear-gradient(135deg, #0d2b1f, #1a3d2b);
    border: 1px solid #27ae60;
    border-left: 5px solid #27ae60;
}
.result-label {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 0;
    color: white;
}
.result-conf {
    font-size: 1rem;
    margin: 0.4rem 0 0 0;
    color: #cbd5e1;
}
.result-note {
    font-size: 0.82rem;
    margin-top: 0.6rem;
}

/* ── Heatmap legend ── */
.heatmap-legend {
    display: flex;
    gap: 1rem;
    margin-top: 0.6rem;
    padding: 0.6rem 1rem;
    background: #1e2533;
    border-radius: 8px;
    font-size: 0.78rem;
    color: #94a3b8;
}

/* ── Stats row ── */
.stat-box {
    background: #1e2533;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
    border: 1px solid #2d3748;
}
.stat-value {
    font-size: 1.4rem;
    font-weight: 800;
    color: #00c9ff;
    margin: 0;
}
.stat-label {
    font-size: 0.72rem;
    color: #64748b;
    margin: 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #1a1f2e;
    border: 2px dashed #2d3748;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Hide deprecation warnings ── */
.stAlert {
    display: none !important;
}
[data-testid="stNotification"] {
    display: none !important;
}
div[data-baseweb="notification"] {
    display: none !important;
}

/* ── Progress bars ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #028090, #00c9ff) !important;
}

/* ── Divider ── */
hr {
    border-color: #2d3748 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="onco-header">
    <div>
        <p class="onco-title">🦷 ONCOAi</p>
        <p class="onco-sub">AI-Powered Oral Cancer Detection System &nbsp;·&nbsp;
        <span class="onco-badge">MobileNetV2 · 90% Accuracy</span></p>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Helper ────────────────────────────────────────────────────────────────────
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "📂  Upload an oral cavity image (JPG / JPEG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

if uploaded is not None:
    image = Image.open(uploaded)

    # ── Run Prediction + Grad-CAM ─────────────────────────────────────────────
    with st.spinner("🔬 Analysing image with AI..."):
        pred_class, confidence, all_probs = predict(image)
        pred_index     = 0 if pred_class == 'CANCER' else 1
        model          = get_model()
        last_conv      = get_last_conv_layer_name(model)
        gradcam_img, _ = generate_gradcam(model, image, pred_index, last_conv)

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value">{confidence:.1f}%</p>
            <p class="stat-label">Confidence</p>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value">{'🚨' if pred_class=='CANCER' else '✅'} {pred_class}</p>
            <p class="stat-label">Prediction</p>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value">{all_probs.get('CANCER', 0):.1f}%</p>
            <p class="stat-label">Cancer Probability</p>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value">{all_probs.get('NON CANCER', 0):.1f}%</p>
            <p class="stat-label">Non-Cancer Probability</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main 3 column layout ──────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown('<p class="section-label">📷 Original Image</p>', unsafe_allow_html=True)
        st.image(image, width=320)

    with col2:
        st.markdown('<p class="section-label">🔥 Grad-CAM Heatmap</p>', unsafe_allow_html=True)
        st.image(gradcam_img, width=320)
        st.markdown("""
        <div class="heatmap-legend">
            <span>🔴 <b>Red/Yellow</b> — Suspicious region</span>
            <span>🔵 <b>Blue/Purple</b> — Normal tissue</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown('<p class="section-label">📊 Diagnosis Result</p>', unsafe_allow_html=True)

        if pred_class == 'CANCER':
            st.markdown(f"""
            <div class="cancer-box">
                <p class="result-label">🚨 CANCER DETECTED</p>
                <p class="result-conf">Confidence Score: <b>{confidence:.1f}%</b></p>
                <p class="result-note" style="color:#f87171;">
                ⚠️ Suspicious lesion detected. Please consult an oncologist or dental specialist immediately for further examination.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <p class="result-label">✅ NO CANCER DETECTED</p>
                <p class="result-conf">Confidence Score: <b>{confidence:.1f}%</b></p>
                <p class="result-note" style="color:#6ee7b7;">
                No malignant lesion detected. Continue regular dental checkups for early detection.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">Probability Breakdown</p>', unsafe_allow_html=True)

        cancer_prob    = all_probs.get('CANCER',     0.0) / 100.0
        noncancer_prob = all_probs.get('NON CANCER', 0.0) / 100.0
        st.progress(float(cancer_prob),    text=f"🔴  CANCER:      {cancer_prob*100:.1f}%")
        st.progress(float(noncancer_prob), text=f"🟢  NON CANCER:  {noncancer_prob*100:.1f}%")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <p style="color:#475569; font-size:0.78rem; text-align:center;">
    ⚠️ <b>Medical Disclaimer:</b> ONCOAi is an AI-assisted screening tool intended to support clinical decision-making.
    It is NOT a substitute for professional medical diagnosis. Always consult a qualified dentist or oncologist
    for clinical evaluation and treatment decisions.
    </p>
    """, unsafe_allow_html=True)

else:
    # ── Landing ───────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#64748b; text-align:center; font-size:1rem;">
    Upload an oral cavity image above to begin analysis
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("1️⃣", "Upload", "Upload a clear JPG/PNG photo of the oral cavity"),
        ("2️⃣", "Analyse", "MobileNetV2 deep learning model processes the image"),
        ("3️⃣", "Predict", "Get CANCER or NON CANCER result with confidence score"),
        ("4️⃣", "Explain", "Grad-CAM heatmap shows the suspicious region visually"),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="stat-box" style="text-align:left; padding:1.2rem;">
                <p style="font-size:1.5rem; margin:0">{icon}</p>
                <p style="font-weight:700; color:white; margin:0.4rem 0 0.2rem 0">{title}</p>
                <p style="font-size:0.82rem; color:#64748b; margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style="font-size:1.3rem; font-weight:800;
    background:linear-gradient(90deg,#00c9ff,#028090);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    🔬 ONCOAi
    </p>
    """, unsafe_allow_html=True)

    st.markdown("**Team MediScope**")
    st.markdown("""
<small>
👩‍💻 Sasmita D — 727823TUCS305<br>
👨‍💻 Sedhupathi R — 727823TUCS308<br>
👨‍💻 Shabin George — 727823TUCS310
</small>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**Mentor:** Dr. Udhayamoorthi M")
    st.divider()

    st.markdown("""
<small>
<b>Model</b><br>MobileNetV2 (Transfer Learning)<br><br>
<b>Pre-trained On</b><br>ImageNet (1.2M images)<br><br>
<b>Dataset</b><br>1700 oral cavity images<br><br>
<b>Input Size</b><br>224 × 224 px<br><br>
<b>Test Accuracy</b><br>90% (F1 = 0.90)<br><br>
<b>Explainability</b><br>Grad-CAM heatmaps<br><br>
<b>Classes</b><br>🚨 CANCER &nbsp;·&nbsp; ✅ NON CANCER
</small>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <small style="color:#475569">
    Sem 6 · Review 2 · April 2026<br>
    Dept. of CSE
    </small>
    """, unsafe_allow_html=True)
