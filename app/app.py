
import streamlit as st
from PIL import Image
import sys, os

sys.path.append(os.path.dirname(__file__))
from predictor import predict

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ONCOAi — Oral Cancer Detection",
    page_icon="🦷",
    layout="centered"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.title      { font-size:2.2rem; font-weight:800; color:#028090; }
.sub        { font-size:1rem;   color:#64748B; margin-bottom:1rem; }
.cancer-box { padding:1rem 1.5rem; border-radius:10px;
              background:#fdecea; border-left:6px solid #e74c3c; }
.safe-box   { padding:1rem 1.5rem; border-radius:10px;
              background:#eafaf1; border-left:6px solid #27ae60; }
.label      { font-size:1.5rem; font-weight:700; margin:0; }
.conf       { font-size:1.1rem; margin:0.3rem 0 0 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="title">🦷 ONCOAi</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">AI-Powered Oral Cancer Detection &nbsp;|&nbsp; Team MediScope</p>',
    unsafe_allow_html=True
)
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an oral cavity image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("🔍 Analysing image..."):
            pred_class, confidence, all_probs = predict(image)

        # all_probs values are already in percentage (0-100)
        # Convert to 0.0-1.0 for display
        cancer_prob    = all_probs.get('CANCER',     0.0) / 100.0
        noncancer_prob = all_probs.get('NON CANCER', 0.0) / 100.0

        # ── Result card ───────────────────────────────────────────────────────
        if pred_class == 'CANCER':
            st.markdown(f"""
            <div class="cancer-box">
                <p class="label">🚨 {pred_class}</p>
                <p class="conf">Confidence: <b>{confidence:.1f}%</b></p>
                <p style="font-size:0.85rem; margin-top:0.5rem; color:#c0392b;">
                Please consult a medical professional immediately.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-box">
                <p class="label">✅ {pred_class}</p>
                <p class="conf">Confidence: <b>{confidence:.1f}%</b></p>
                <p style="font-size:0.85rem; margin-top:0.5rem; color:#1e8449;">
                No signs of cancer detected. Continue regular checkups.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability bars ──────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("**Prediction Breakdown**")

        st.progress(
            float(cancer_prob),
            text=f"CANCER: {cancer_prob * 100:.1f}%"
        )
        st.progress(
            float(noncancer_prob),
            text=f"NON CANCER: {noncancer_prob * 100:.1f}%"
        )

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ ONCOAi is an AI screening tool — NOT a medical diagnosis. "
        "Always consult a qualified dentist or oncologist."
    )

else:
    # Placeholder when nothing uploaded yet
    st.info("👆 Upload an oral cavity image above to get a prediction")

    st.markdown("#### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1️⃣ Upload**\nUpload a clear photo of the oral cavity")
    with col2:
        st.markdown("**2️⃣ Analyse**\nEfficientNetB0 model analyses the image")
    with col3:
        st.markdown("**3️⃣ Result**\nGet prediction with confidence score")

