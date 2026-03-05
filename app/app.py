
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
    layout="wide"
)

# ── SVG Icons ─────────────────────────────────────────────────────────────────
ICONS = {
    "microscope": '<svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="#028090" viewBox="0 0 512 512"><path d="M160 96a96 96 0 1 1 192 0A96 96 0 1 1 160 96zM144 480l-48 0c-17.7 0-32-14.3-32-32s14.3-32 32-32l48 0 0-32-48 0c-53 0-96 43-96 96s43 96 96 96l304 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-256 0 0-32zm176-96l-48 0 0 96 48 0 0-96zM256 320l0-32-48 0 0 32 48 0zm64 0l48 0 0-32-48 0 0 32z"/></svg>',
    "upload":     '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#028090" viewBox="0 0 640 512"><path d="M144 480C64.5 480 0 415.5 0 336c0-62.8 40.2-116.2 96.2-135.9c-.1-2.7-.2-5.4-.2-8.1c0-88.4 71.6-160 160-160c59.3 0 111 32.2 138.7 80.2C409.9 102 428.3 96 448 96c53 0 96 43 96 96c0 12.2-2.3 23.8-6.4 34.6C596 238.4 640 290.1 640 352c0 70.7-57.3 128-128 128H144zm79-217c-9.4 9.4-9.4 24.6 0 33.9s24.6 9.4 33.9 0l39-39V392c0 13.3 10.7 24 24 24s24-10.7 24-24V257.9l39 39c9.4 9.4 24.6 9.4 33.9 0s9.4-24.6 0-33.9l-80-80c-9.4-9.4-24.6-9.4-33.9 0l-80 80z"/></svg>',
    "brain":      '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="#028090" viewBox="0 0 512 512"><path d="M184 0c30.9 0 56 25.1 56 56l0 400c0 30.9-25.1 56-56 56c-28.9 0-52.7-21.9-55.7-50.1C126.9 453.3 126 443.8 126 432c-24.3 5.4-51 2.3-72.3-9.3C37.7 414.2 26.5 397.7 26.5 381c0-4 .6-7.9 1.7-11.6C12.5 356.3 0 338.4 0 317c0-17.5 8.1-33.1 20.8-43.5C7.4 263 0 247.8 0 231c0-19 9.4-35.9 23.8-46.3C18.5 175.2 16 165.4 16 155c0-30.9 25.1-56 56-56c8.6 0 16.8 1.9 24.1 5.4C106.6 80.9 137.8 56 176 56c3.3 0 6.5 .2 9.7 .5C184.8 57.5 184 56.8 184 56c0-30.9 25.1-56 56-56h-56zm144 0c30.9 0 56 25.1 56 56c0 .8-.8 1.5-.7 2.5C386.5 56.2 389.7 56 393 56c38.2 0 69.4 24.9 79.9 59.4c7.3-3.5 15.5-5.4 24.1-5.4c30.9 0 56 25.1 56 56c0 10.4-2.5 20.2-7.8 29.7C559.4 206.5 568 222.6 568 243c0 16.8-7.4 32-20.8 42.5C560.6 296.1 568 313 568 331c0 21.4-12.5 39.3-28.5 52.4c1.1 3.7 1.7 7.6 1.7 11.6c0 16.7-11.2 33.2-27.2 41.7C492 448.3 465.3 451.4 441 446c0 11.8-.9 21.3-2.3 29.9C435.7 490.1 411.9 512 383 512c-30.9 0-56-25.1-56-56l0-400c0-30.9 25.1-56 56-56h-55z"/></svg>',
    "stethoscope":'<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="#028090" viewBox="0 0 576 512"><path d="M142.4 21.9c5.6 16.8-3.5 34.9-20.2 40.5L96 71.1 96 192c0 53 43 96 96 96s96-43 96-96l0-120.9-26.1-8.7c-16.8-5.6-25.8-23.7-20.2-40.5s23.7-25.8 40.5-20.2l26.1 8.7C334.4 19.1 352 43.5 352 71.1L352 192c0 77.2-54.6 141.6-127.3 156.7C231 404.6 278.4 448 336 448c61.9 0 112-50.1 112-112l0-70.7c-28.3-12.3-48-40.5-48-73.3c0-44.2 35.8-80 80-80s80 35.8 80 80c0 32.8-19.7 61-48 73.3l0 70.7c0 97.2-78.8 176-176 176c-92.9 0-168.9-71.9-175.5-163.1C87.2 334.2 32 269.6 32 192L32 71.1c0-27.5 17.6-52 43.9-60.4l26.1-8.7c16.8-5.6 34.9 3.5 40.5 20.2zM480 224a32 32 0 1 0 0-64 32 32 0 1 0 0 64z"/></svg>',
    "fire":       '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" fill="#e74c3c" viewBox="0 0 448 512"><path d="M159.3 5.4c7.8-7.3 19.9-7.2 27.7 .1c27.6 25.9 53.5 53.8 77.7 84c11-14.4 23.5-30.1 37-42.9c7.9-7.4 20.1-7.4 28 .1c34.6 33 63.9 76.6 84.5 118c20.3 40.8 33.8 82.5 33.8 111.9C448 404.2 348.2 512 224 512C99.8 512 0 404.2 0 276.5c0-38.4 17.8-85.3 45.4-131.7C73.3 97.7 112.7 48.6 159.3 5.4zM225.7 416c25.3 0 47.7-7 68.8-21c42.1-29.4 53.4-88.2 28.1-134.4c-4.5-9-16-9.6-22.5-2l-25.2 29.3c-6.6 7.6-18.5 7.4-24.9-.3l-6.1-7.1c-7.4-8.5-18.5-10.2-27.6-5.7C170.3 288.5 160 307.7 160 329c0 48.6 31.3 87 65.7 87z"/></svg>',
    "chart":      '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#028090" viewBox="0 0 448 512"><path d="M160 80c0-26.5 21.5-48 48-48l32 0c26.5 0 48 21.5 48 48l0 352c0 26.5-21.5 48-48 48l-32 0c-26.5 0-48-21.5-48-48l0-352zM0 272c0-26.5 21.5-48 48-48l32 0c26.5 0 48 21.5 48 48l0 160c0 26.5-21.5 48-48 48l-32 0c-26.5 0-48-21.5-48-48L0 272zM368 96l32 0c26.5 0 48 21.5 48 48l0 288c0 26.5-21.5 48-48 48l-32 0c-26.5 0-48-21.5-48-48l0-288c0-26.5 21.5-48 48-48z"/></svg>',
    "warning":    '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#e74c3c" viewBox="0 0 512 512"><path d="M256 32c14.2 0 27.3 7.5 34.5 19.8l216 368c7.3 12.4 7.3 27.7 .2 40.1S486.3 480 472 480L40 480c-14.3 0-27.6-7.7-34.7-20.1s-7-27.8 .2-40.1l216-368C228.7 39.5 241.8 32 256 32zm0 128c-13.3 0-24 10.7-24 24l0 112c0 13.3 10.7 24 24 24s24-10.7 24-24l0-112c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0 -64 0 32 32 0 1 0 64 0z"/></svg>',
    "check":      '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" fill="#27ae60" viewBox="0 0 512 512"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM369 209L241 337c-9.4 9.4-24.6 9.4-33.9 0l-64-64c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l47 47L335 175c9.4-9.4 24.6-9.4 33.9 0s9.4 24.6 0 33.9z"/></svg>',
    "hospital":   '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#f87171" viewBox="0 0 576 512"><path d="M0 96l576 0c0-35.3-28.7-64-64-64L64 32C28.7 32 0 60.7 0 96zm0 32L0 416c0 35.3 28.7 64 64 64l448 0c35.3 0 64-28.7 64-64l0-288L0 128zM240 272l0-48 48 0 0-48 48 0 0 48 48 0 0 48-48 0 0 48-48 0 0-48-48 0z"/></svg>',
    "calendar":   '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#6ee7b7" viewBox="0 0 448 512"><path d="M128 0c17.7 0 32 14.3 32 32l0 32 128 0 0-32c0-17.7 14.3-32 32-32s32 14.3 32 32l0 32 48 0c26.5 0 48 21.5 48 48l0 48L0 160l0-48C0 85.5 21.5 64 48 64l48 0 0-32c0-17.7 14.3-32 32-32zM0 192l448 0 0 272c0 26.5-21.5 48-48 48L48 512c-26.5 0-48-21.5-48-48L0 192zm64 80l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0c-8.8 0-16 7.2-16 16zm128 0l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0c-8.8 0-16 7.2-16 16zm144-16c-8.8 0-16 7.2-16 16l0 32c0 8.8 7.2 16 16 16l32 0c8.8 0 16-7.2 16-16l0-32c0-8.8-7.2-16-16-16l-32 0z"/></svg>',
    "user":       '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="#028090" viewBox="0 0 448 512"><path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512l388.6 0c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304l-91.4 0z"/></svg>',
    "gauge":      '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#028090" viewBox="0 0 512 512"><path d="M0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256zm320 96c0-26.9-16.5-49.9-40-59.3L280 88c0-13.3-10.7-24-24-24s-24 10.7-24 24l0 204.7c-23.5 9.5-40 32.5-40 59.3c0 35.3 28.7 64 64 64s64-28.7 64-64z"/></svg>',
    "biohazard":  '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#e74c3c" viewBox="0 0 576 512"><path d="M287.9 112c-74.6 0-135.4 60.8-135.4 135.4c0 37.4 15.2 71.2 39.7 95.7l-87.9 87.9C67.6 394.9 47.9 350.6 47.9 300.9c0-90.4 54.9-168.2 134.1-200.3C166.8 81.9 152.9 57.6 148.7 30C134.3 11.4 112.5 0 87.9 0C39.3 0 0 39.3 0 87.9c0 28.5 13.5 53.8 34.5 70c-21 32.1-34.5 69.8-34.5 110.8c0 84.6 50.4 157.8 123.2 190.4L88.4 494c-6.3 6.3-6.3 16.4 0 22.6s16.4 6.3 22.6 0L256 371.7l145 145c6.3 6.3 16.4 6.3 22.6 0s6.3-16.4 0-22.6l-34.8-34.8C461.6 426.5 512 353.4 512 268.7c0-41-13.5-78.7-34.5-110.8c21-16.2 34.5-41.5 34.5-70C512 39.3 472.7 0 424.1 0c-24.6 0-46.4 11.4-60.8 30c-4.2 27.6-18.1 51.9-33.3 70.6C352.8 132.7 320.4 112 287.9 112z"/></svg>',
    "shield":     '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#27ae60" viewBox="0 0 512 512"><path d="M269.4 2.9C265.2 1 260.7 0 256 0s-9.2 1-13.4 2.9L54.3 82.8c-22 9.3-38.4 31-38.3 57.2c.5 99.2 41.3 280.7 213.6 363.2c16.7 8 36.1 8 52.8 0C454.7 420.7 495.5 239.2 496 140c.1-26.2-16.3-47.9-38.3-57.2L269.4 2.9z"/></svg>',
    "info":       '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#028090" viewBox="0 0 512 512"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216 336l24 0 0-64-24 0c-13.3 0-24-10.7-24-24s10.7-24 24-24l48 0c13.3 0 24 10.7 24 24l0 88 8 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-80 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32 32 0 1 1 0 64 32 32 0 1 1 0-64z"/></svg>',
}

def icon(name):
    return ICONS.get(name, "")

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
.onco-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00c9ff, #028090);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    display: inline-block;
}
.onco-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0.3rem 0 0 0;
}
.onco-badge {
    background: #028090;
    color: white;
    padding: 0.25rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #94a3b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Stat boxes ── */
.stat-box {
    background: #1e2533;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    border: 1px solid #2d3748;
}
.stat-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #00c9ff;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}
.stat-label {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0.3rem 0 0 0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.4rem;
}

/* ── Result cards ── */
.cancer-box {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #2d1515, #3d1f1f);
    border: 1px solid #e74c3c;
    border-left: 6px solid #e74c3c;
}
.safe-box {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #0d2b1f, #1a3d2b);
    border: 1px solid #27ae60;
    border-left: 6px solid #27ae60;
}
.result-label {
    font-size: 1.7rem;
    font-weight: 800;
    margin: 0;
    color: white;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.result-conf {
    font-size: 1.1rem;
    margin: 0.5rem 0 0 0;
    color: #cbd5e1;
}
.result-note {
    font-size: 0.9rem;
    margin-top: 0.7rem;
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    line-height: 1.5;
}

/* ── Heatmap legend ── */
.heatmap-legend {
    display: flex;
    gap: 1.2rem;
    margin-top: 0.7rem;
    padding: 0.7rem 1rem;
    background: #1e2533;
    border-radius: 8px;
    font-size: 0.86rem;
    color: #94a3b8;
    align-items: center;
}

/* ── How it works cards ── */
.how-card {
    background: #1e2533;
    border-radius: 12px;
    padding: 1.4rem;
    border: 1px solid #2d3748;
    height: 100%;
    transition: border-color 0.2s;
}
.how-card:hover {
    border-color: #028090;
}
.how-card h4 {
    color: white;
    margin: 0.6rem 0 0.4rem 0;
    font-size: 1.05rem;
    font-weight: 700;
}
.how-card p {
    color: #64748b;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.5;
}

/* ── Sidebar ── */
.sidebar-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0.4rem 0;
    line-height: 1.6;
}
.sidebar-key {
    color: #94a3b8;
    font-weight: 600;
}
.sidebar-val {
    color: #64748b;
}

/* ── Hide Streamlit warnings ── */
.stAlert                        { display: none !important; }
[data-testid="stNotification"]  { display: none !important; }
div[data-baseweb="notification"]{ display: none !important; }

/* ── Progress bars ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #028090, #00c9ff) !important;
    border-radius: 4px !important;
}

/* ── Divider ── */
hr { border-color: #2d3748 !important; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #1a1f2e;
    border: 2px dashed #2d3748;
    border-radius: 12px;
    padding: 0.5rem;
}

</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:1.5rem 0 0.5rem 0;">
    <div style="display:flex; align-items:center; gap:0.8rem;">
        {icon("microscope")}
        <p class="onco-title">ONCOAi</p>
    </div>
    <p class="onco-sub">
        AI-Powered Oral Cancer Detection System &nbsp;&middot;&nbsp;
        <span class="onco-badge">MobileNetV2 &nbsp;&middot;&nbsp; 92.4% Accuracy</span>
    </p>
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
st.markdown(f"""
<p style="color:#94a3b8; font-size:0.95rem; margin-bottom:0.4rem;
display:flex; align-items:center; gap:0.5rem;">
    {icon("upload")}
    <span>Upload an oral cavity image &nbsp;(JPG / JPEG / PNG)</span>
</p>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:
# ══════════════════════════════════════════════════════════════════════════════

    image = Image.open(uploaded)

    with st.spinner("Analysing image with AI..."):
        pred_class, confidence, all_probs = predict(image)
        pred_index     = 0 if pred_class == 'CANCER' else 1
        model          = get_model()
        last_conv      = get_last_conv_layer_name(model)
        gradcam_img, _ = generate_gradcam(model, image, pred_index, last_conv)

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)

    result_color = '#e74c3c' if pred_class == 'CANCER' else '#27ae60'
    result_icon  = icon("warning") if pred_class == 'CANCER' else icon("check")

    with s1:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value">{confidence:.1f}%</p>
            <p class="stat-label">{icon("gauge")} Confidence Score</p>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value" style="color:{result_color};">
                {result_icon} {pred_class}
            </p>
            <p class="stat-label">{icon("stethoscope")} Prediction</p>
        </div>""", unsafe_allow_html=True)

    with s3:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value" style="color:#e74c3c;">
                {all_probs.get('CANCER', 0):.1f}%
            </p>
            <p class="stat-label">{icon("biohazard")} Cancer Probability</p>
        </div>""", unsafe_allow_html=True)

    with s4:
        st.markdown(f"""
        <div class="stat-box">
            <p class="stat-value" style="color:#27ae60;">
                {all_probs.get('NON CANCER', 0):.1f}%
            </p>
            <p class="stat-label">{icon("shield")} Non-Cancer Probability</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 3 column layout ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown(f"""
        <p class="section-label">{icon("upload")} Original Image</p>
        """, unsafe_allow_html=True)
        st.image(image, width=320)

    with col2:
        st.markdown(f"""
        <p class="section-label">{icon("fire")} Grad-CAM Heatmap</p>
        """, unsafe_allow_html=True)
        st.image(gradcam_img, width=320)
        st.markdown("""
        <div class="heatmap-legend">
            <span style="display:flex; align-items:center; gap:0.5rem;">
                <svg width="12" height="12">
                    <circle cx="6" cy="6" r="6" fill="#e74c3c"/>
                </svg>
                Red / Yellow — Suspicious region
            </span>
            <span style="display:flex; align-items:center; gap:0.5rem;">
                <svg width="12" height="12">
                    <circle cx="6" cy="6" r="6" fill="#3498db"/>
                </svg>
                Blue / Purple — Normal tissue
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <p class="section-label">{icon("chart")} Diagnosis Result</p>
        """, unsafe_allow_html=True)

        if pred_class == 'CANCER':
            st.markdown(f"""
            <div class="cancer-box">
                <p class="result-label">
                    {icon("warning")} CANCER DETECTED
                </p>
                <p class="result-conf">
                    Confidence: <b>{confidence:.1f}%</b>
                </p>
                <p class="result-note" style="color:#f87171;">
                    {icon("hospital")}
                    Suspicious lesion detected. Please consult an oncologist
                    or dental specialist immediately for further examination.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <p class="result-label">
                    {icon("check")} NO CANCER DETECTED
                </p>
                <p class="result-conf">
                    Confidence: <b>{confidence:.1f}%</b>
                </p>
                <p class="result-note" style="color:#6ee7b7;">
                    {icon("calendar")}
                    No malignant lesion detected. Continue regular dental
                    checkups for early detection.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <p class="section-label">{icon("chart")} Probability Breakdown</p>
        """, unsafe_allow_html=True)

        cancer_prob    = all_probs.get('CANCER',     0.0) / 100.0
        noncancer_prob = all_probs.get('NON CANCER', 0.0) / 100.0
        st.progress(float(cancer_prob),    text=f"CANCER:      {cancer_prob * 100:.1f}%")
        st.progress(float(noncancer_prob), text=f"NON CANCER:  {noncancer_prob * 100:.1f}%")

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(f"""
    <p style="color:#475569; font-size:0.82rem; text-align:center;
    display:flex; align-items:center; justify-content:center; gap:0.5rem;">
        {icon("info")}
        <span>
            <b>Medical Disclaimer:</b> ONCOAi is an AI-assisted screening tool
            intended to support clinical decision-making. It is NOT a substitute
            for professional medical diagnosis. Always consult a qualified
            dentist or oncologist for evaluation and treatment decisions.
        </span>
    </p>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
else:
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <p style="color:#64748b; text-align:center; font-size:1.05rem;">
        Upload an oral cavity image above to begin AI analysis
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("upload",      "Upload",   "Upload a clear JPG or PNG photo of the oral cavity"),
        ("brain",       "Analyse",  "MobileNetV2 deep learning model processes the image"),
        ("stethoscope", "Predict",  "Get CANCER or NON CANCER result with confidence score"),
        ("fire",        "Explain",  "Grad-CAM heatmap visually highlights the suspicious region"),
    ]
    for col, (ico, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="how-card">
                {icon(ico)}
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
        {icon("microscope")}
        <span style="font-size:1.4rem; font-weight:800;
        background:linear-gradient(90deg,#00c9ff,#028090);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        ONCOAi</span>
    </div>
    <p style="color:#94a3b8; font-size:0.9rem; font-weight:600; margin:0 0 0.5rem 0;">
        Team MediScope
    </p>
    <div style="font-size:0.88rem; line-height:2.1;">
        <div class="sidebar-row">{icon("user")} Sasmita D &nbsp;— 727823TUCS305</div>
        <div class="sidebar-row">{icon("user")} Sedhupathi R — 727823TUCS308</div>
        <div class="sidebar-row">{icon("user")} Shabin George — 727823TUCS310</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(f"""
    <div class="sidebar-row">
        {icon("user")}
        <span>
            <span class="sidebar-key">Mentor: </span>
            <span class="sidebar-val">Dr. Udhayamoorthi M</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    model_rows = [
        ("brain",       "Model",          "MobileNetV2"),
        ("chart",       "Pre-trained On", "ImageNet"),
        ("upload",      "Dataset",        "1700 images"),
        ("gauge",       "Input Size",     "224 × 224 px"),
        ("chart",       "Test Accuracy",  "92.4%  (F1 = 0.92)"),
        ("fire",        "Explainability", "Grad-CAM"),
        ("stethoscope", "Classes",        "CANCER / NON CANCER"),
    ]
    rows_html = ""
    for ico, key, val in model_rows:
        rows_html += f"""
        <div class="sidebar-row">
            {icon(ico)}
            <span>
                <span class="sidebar-key">{key}: </span>
                <span class="sidebar-val">{val}</span>
            </span>
        </div>"""
    st.markdown(f'<div style="font-size:0.88rem; line-height:2.2;">{rows_html}</div>',
                unsafe_allow_html=True)

    st.divider()

    st.markdown(f"""
    <div style="font-size:0.82rem; line-height:2.2;">
            {icon("hospital")}
            <span class="sidebar-val">Dept. of Computer Science &amp; Engg.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
