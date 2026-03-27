
import streamlit as st
from PIL import Image
import sys, os
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))
from predictor import predict, get_model
from gradcam import generate_gradcam
from auth import show_auth_page, show_user_badge
from chatbot import initialize_chatbot_session, render_simple_chat, render_quick_links

st.set_page_config(
    page_title="ONCOAi — Oral Cancer Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Auth gate — show login page if not logged in ──────────────────────────────
if not show_auth_page():
    st.stop()

# Show user badge in sidebar
show_user_badge()


# ── SVG Icons ─────────────────────────────────────────────────────────────────
ICONS = {
    "microscope": '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M160 96a96 96 0 1 1 192 0A96 96 0 1 1 160 96zM144 480l-48 0c-17.7 0-32-14.3-32-32s14.3-32 32-32l48 0 0-32-48 0c-53 0-96 43-96 96s43 96 96 96l304 0c17.7 0 32-14.3 32-32s-14.3-32-32-32l-256 0 0-32zm176-96l-48 0 0 96 48 0 0-96zM256 320l0-32-48 0 0 32 48 0zm64 0l48 0 0-32-48 0 0 32z"/></svg>',
    "upload":     '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#1a6b5e" viewBox="0 0 640 512"><path d="M144 480C64.5 480 0 415.5 0 336c0-62.8 40.2-116.2 96.2-135.9c-.1-2.7-.2-5.4-.2-8.1c0-88.4 71.6-160 160-160c59.3 0 111 32.2 138.7 80.2C409.9 102 428.3 96 448 96c53 0 96 43 96 96c0 12.2-2.3 23.8-6.4 34.6C596 238.4 640 290.1 640 352c0 70.7-57.3 128-128 128H144zm79-217c-9.4 9.4-9.4 24.6 0 33.9s24.6 9.4 33.9 0l39-39V392c0 13.3 10.7 24 24 24s24-10.7 24-24V257.9l39 39c9.4 9.4 24.6 9.4 33.9 0s9.4-24.6 0-33.9l-80-80c-9.4-9.4-24.6-9.4-33.9 0l-80 80z"/></svg>',
    "brain":      '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M184 0c30.9 0 56 25.1 56 56l0 400c0 30.9-25.1 56-56 56c-28.9 0-52.7-21.9-55.7-50.1C126.9 453.3 126 443.8 126 432c-24.3 5.4-51 2.3-72.3-9.3C37.7 414.2 26.5 397.7 26.5 381c0-4 .6-7.9 1.7-11.6C12.5 356.3 0 338.4 0 317c0-17.5 8.1-33.1 20.8-43.5C7.4 263 0 247.8 0 231c0-19 9.4-35.9 23.8-46.3C18.5 175.2 16 165.4 16 155c0-30.9 25.1-56 56-56c8.6 0 16.8 1.9 24.1 5.4C106.6 80.9 137.8 56 176 56c3.3 0 6.5.2 9.7.5C184.8 57.5 184 56.8 184 56c0-30.9 25.1-56 56-56h-56zm144 0c30.9 0 56 25.1 56 56c0 .8-.8 1.5-.7 2.5C386.5 56.2 389.7 56 393 56c38.2 0 69.4 24.9 79.9 59.4c7.3-3.5 15.5-5.4 24.1-5.4c30.9 0 56 25.1 56 56c0 10.4-2.5 20.2-7.8 29.7C559.4 206.5 568 222.6 568 243c0 16.8-7.4 32-20.8 42.5C560.6 296.1 568 313 568 331c0 21.4-12.5 39.3-28.5 52.4c1.1 3.7 1.7 7.6 1.7 11.6c0 16.7-11.2 33.2-27.2 41.7C492 448.3 465.3 451.4 441 446c0 11.8-.9 21.3-2.3 29.9C435.7 490.1 411.9 512 383 512c-30.9 0-56-25.1-56-56l0-400c0-30.9 25.1-56 56-56h-55z"/></svg>',
    "stethoscope":'<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="#1a6b5e" viewBox="0 0 576 512"><path d="M142.4 21.9c5.6 16.8-3.5 34.9-20.2 40.5L96 71.1 96 192c0 53 43 96 96 96s96-43 96-96l0-120.9-26.1-8.7c-16.8-5.6-25.8-23.7-20.2-40.5s23.7-25.8 40.5-20.2l26.1 8.7C334.4 19.1 352 43.5 352 71.1L352 192c0 77.2-54.6 141.6-127.3 156.7C231 404.6 278.4 448 336 448c61.9 0 112-50.1 112-112l0-70.7c-28.3-12.3-48-40.5-48-73.3c0-44.2 35.8-80 80-80s80 35.8 80 80c0 32.8-19.7 61-48 73.3l0 70.7c0 97.2-78.8 176-176 176c-92.9 0-168.9-71.9-175.5-163.1C87.2 334.2 32 269.6 32 192L32 71.1c0-27.5 17.6-52 43.9-60.4l26.1-8.7c16.8-5.6 34.9 3.5 40.5 20.2zM480 224a32 32 0 1 0 0-64 32 32 0 1 0 0 64z"/></svg>',
    "fire":       '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#c0392b" viewBox="0 0 448 512"><path d="M159.3 5.4c7.8-7.3 19.9-7.2 27.7.1c27.6 25.9 53.5 53.8 77.7 84c11-14.4 23.5-30.1 37-42.9c7.9-7.4 20.1-7.4 28 .1c34.6 33 63.9 76.6 84.5 118c20.3 40.8 33.8 82.5 33.8 111.9C448 404.2 348.2 512 224 512C99.8 512 0 404.2 0 276.5c0-38.4 17.8-85.3 45.4-131.7C73.3 97.7 112.7 48.6 159.3 5.4z"/></svg>',
    "chart":      '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#1a6b5e" viewBox="0 0 448 512"><path d="M160 80c0-26.5 21.5-48 48-48l32 0c26.5 0 48 21.5 48 48l0 352c0 26.5-21.5 48-48 48l-32 0c-26.5 0-48-21.5-48-48l0-352zM0 272c0-26.5 21.5-48 48-48l32 0c26.5 0 48 21.5 48 48l0 160c0 26.5-21.5 48-48 48l-32 0c-26.5 0-48-21.5-48-48L0 272zM368 96l32 0c26.5 0 48 21.5 48 48l0 288c0 26.5-21.5 48-48 48l-32 0c-26.5 0-48-21.5-48-48l0-288c0-26.5 21.5-48 48-48z"/></svg>',
    "warning":    '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#c0392b" viewBox="0 0 512 512"><path d="M256 32c14.2 0 27.3 7.5 34.5 19.8l216 368c7.3 12.4 7.3 27.7.2 40.1S486.3 480 472 480L40 480c-14.3 0-27.6-7.7-34.7-20.1s-7-27.8.2-40.1l216-368C228.7 39.5 241.8 32 256 32zm0 128c-13.3 0-24 10.7-24 24l0 112c0 13.3 10.7 24 24 24s24-10.7 24-24l0-112c0-13.3-10.7-24-24-24zm32 224a32 32 0 1 0-64 0 32 32 0 1 0 64 0z"/></svg>',
    "suspicious": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#d68910" viewBox="0 0 512 512"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zm0-384c13.3 0 24 10.7 24 24l0 112c0 13.3-10.7 24-24 24s-24-10.7-24-24l0-112c0-13.3 10.7-24 24-24zm-32 224a32 32 0 1 1 64 0 32 32 0 1 1-64 0z"/></svg>',
    "check":      '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM369 209L241 337c-9.4 9.4-24.6 9.4-33.9 0l-64-64c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l47 47L335 175c9.4-9.4 24.6-9.4 33.9 0s9.4 24.6 0 33.9z"/></svg>',
    "hospital":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#c0392b" viewBox="0 0 576 512"><path d="M0 96l576 0c0-35.3-28.7-64-64-64L64 32C28.7 32 0 60.7 0 96zm0 32L0 416c0 35.3 28.7 64 64 64l448 0c35.3 0 64-28.7 64-64l0-288L0 128zM240 272l0-48 48 0 0-48 48 0 0 48 48 0 0 48-48 0 0 48-48 0 0-48-48 0z"/></svg>',
    "calendar":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#1a6b5e" viewBox="0 0 448 512"><path d="M128 0c17.7 0 32 14.3 32 32l0 32 128 0 0-32c0-17.7 14.3-32 32-32s32 14.3 32 32l0 32 48 0c26.5 0 48 21.5 48 48l0 48L0 160l0-48C0 85.5 21.5 64 48 64l48 0 0-32c0-17.7 14.3-32 32-32zM0 192l448 0 0 272c0 26.5-21.5 48-48 48L48 512c-26.5 0-48-21.5-48-48L0 192z"/></svg>',
    "map":        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="#ffffff" viewBox="0 0 576 512"><path d="M384 476.1L192 421.2l0-385.3L384 90.8l0 385.3zm32-1.2l0-385.3 149.3-55.6C578.6 29 592 40.7 592 56l0 354.5c0 11.8-7.3 22.3-18.3 26.5L416 474.9zM15.7 35.3L160 90.8l0 385.3L32.7 430.7C21.7 426.5 14.3 416 14.3 404.2L14.3 49.7c0-6.8 3.4-13.2 9-17.2c3.9-2.7 8.8-4 13.7-3.7c-.9-.4-1.3.5-.6.5c1.2 0 2.4-.2 3.6-.5c1.2-.3 2.3-.7 3.4-1.2z"/></svg>',
    "user":       '<svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" fill="#1a6b5e" viewBox="0 0 448 512"><path d="M224 256A128 128 0 1 0 224 0a128 128 0 1 0 0 256zm-45.7 48C79.8 304 0 383.8 0 482.3C0 498.7 13.3 512 29.7 512l388.6 0c16.4 0 29.7-13.3 29.7-29.7C448 383.8 368.2 304 269.7 304l-91.4 0z"/></svg>',
    "gauge":      '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M0 256a256 256 0 1 1 512 0A256 256 0 1 1 0 256zm320 96c0-26.9-16.5-49.9-40-59.3L280 88c0-13.3-10.7-24-24-24s-24 10.7-24 24l0 204.7c-23.5 9.5-40 32.5-40 59.3c0 35.3 28.7 64 64 64s64-28.7 64-64z"/></svg>',
    "biohazard":  '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" fill="#c0392b" viewBox="0 0 576 512"><path d="M287.9 112c-74.6 0-135.4 60.8-135.4 135.4c0 37.4 15.2 71.2 39.7 95.7l-87.9 87.9C67.6 394.9 47.9 350.6 47.9 300.9c0-90.4 54.9-168.2 134.1-200.3C166.8 81.9 152.9 57.6 148.7 30C134.3 11.4 112.5 0 87.9 0C39.3 0 0 39.3 0 87.9c0 28.5 13.5 53.8 34.5 70c-21 32.1-34.5 69.8-34.5 110.8c0 84.6 50.4 157.8 123.2 190.4L88.4 494c-6.3 6.3-6.3 16.4 0 22.6s16.4 6.3 22.6 0L256 371.7l145 145c6.3 6.3 16.4 6.3 22.6 0s6.3-16.4 0-22.6l-34.8-34.8C461.6 426.5 512 353.4 512 268.7c0-41-13.5-78.7-34.5-110.8c21-16.2 34.5-41.5 34.5-70C512 39.3 472.7 0 424.1 0c-24.6 0-46.4 11.4-60.8 30c-4.2 27.6-18.1 51.9-33.3 70.6C352.8 132.7 320.4 112 287.9 112z"/></svg>',
    "shield":     '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M269.4 2.9C265.2 1 260.7 0 256 0s-9.2 1-13.4 2.9L54.3 82.8c-22 9.3-38.4 31-38.3 57.2c.5 99.2 41.3 280.7 213.6 363.2c16.7 8 36.1 8 52.8 0C454.7 420.7 495.5 239.2 496 140c.1-26.2-16.3-47.9-38.3-57.2L269.4 2.9z"/></svg>',
    "info":       '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" fill="#1a6b5e" viewBox="0 0 512 512"><path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216 336l24 0 0-64-24 0c-13.3 0-24-10.7-24-24s10.7-24 24-24l48 0c13.3 0 24 10.7 24 24l0 88 8 0c13.3 0 24 10.7 24 24s-10.7 24-24 24l-80 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32 32 0 1 1 0 64 32 32 0 1 1 0-64z"/></svg>',
    "location":   '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="#ffffff" viewBox="0 0 384 512"><path d="M215.7 499.2C267 435 384 279.4 384 192C384 86 298 0 192 0S0 86 0 192c0 87.4 117 243 168.3 307.2c12.3 15.3 35.1 15.3 47.4 0zM192 128a64 64 0 1 1 0 128 64 64 0 1 1 0-128z"/></svg>',
}

def icon(name):
    return ICONS.get(name, "")

def get_diagnosis(all_probs):
    cancer_prob = all_probs.get('CANCER', 0.0)
    if cancer_prob >= 60.0:
        return 'CANCER', cancer_prob
    elif cancer_prob >= 30.0:
        return 'SUSPICIOUS', cancer_prob
    else:
        return 'NON CANCER', all_probs.get('NON CANCER', 0.0)

def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# ── Map HTML component — uses browser geolocation ─────────────────────────────
MAP_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * { margin:0; padding:0; box-sizing:border-box; font-family:'DM Sans',sans-serif; }
  body { background:#f5f7f6; }

  .map-wrapper {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8e6;
    box-shadow: 0 2px 12px rgba(26,107,94,0.10);
  }

  .map-topbar {
    background: #c0392b;
    color: white;
    padding: 0.7rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .map-topbar-left { display:flex; align-items:center; gap:0.5rem; }
  .map-topbar-right {
    font-size:0.72rem; opacity:0.85; font-weight:400;
  }

  #status {
    background: #fff5f5;
    border-bottom: 1px solid #f5c6c6;
    padding: 0.6rem 1.2rem;
    font-size: 0.8rem;
    color: #7b241c;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .dot {
    width:8px; height:8px; border-radius:50%;
    background:#c0392b; animation: pulse 1.2s infinite;
  }
  @keyframes pulse {
    0%,100%{ opacity:1; transform:scale(1); }
    50%    { opacity:0.4; transform:scale(1.3); }
  }

  #map-frame {
    width: 100%;
    height: 420px;
    border: none;
    display: block;
  }

  .map-footer {
    background: #fafcfb;
    border-top: 1px solid #e8f0ee;
    padding: 0.5rem 1.2rem;
    font-size: 0.72rem;
    color: #7a9e97;
    display: flex;
    align-items: center;
    gap: 0.4rem;
  }
</style>
</head>
<body>

<div class="map-wrapper">

  <!-- Top bar -->
  <div class="map-topbar">
    <div class="map-topbar-left">
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
           fill="white" viewBox="0 0 384 512">
        <path d="M215.7 499.2C267 435 384 279.4 384 192C384 86 298 0 192
                 0S0 86 0 192c0 87.4 117 243 168.3 307.2c12.3 15.3 35.1
                 15.3 47.4 0zM192 128a64 64 0 1 1 0 128 64 64 0 1 1 0-128z"/>
      </svg>
      Nearby Cancer &amp; Oncology Hospitals
    </div>
    <span class="map-topbar-right">Powered by Google Maps</span>
  </div>

  <!-- Status bar -->
  <div id="status">
    <div class="dot"></div>
    <span id="status-text">Detecting your location...</span>
  </div>

  <!-- Map iframe -->
  <iframe id="map-frame"
    src="about:blank"
    allowfullscreen
    loading="lazy"
    referrerpolicy="no-referrer-when-downgrade">
  </iframe>

  <!-- Footer -->
  <div class="map-footer">
    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12"
         fill="#7a9e97" viewBox="0 0 512 512">
      <path d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216
               336l24 0 0-64-24 0c-13.3 0-24-10.7-24-24s10.7-24 24-24l48
               0c13.3 0 24 10.7 24 24l0 88 8 0c13.3 0 24 10.7 24 24s-10.7
               24-24 24l-80 0c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32
               32 0 1 1 0 64 32 32 0 1 1 0-64z"/>
    </svg>
    Showing cancer hospitals, oncology centres and dental oncology specialists near you
  </div>

</div>

<script>
  const statusEl = document.getElementById('status-text');
  const mapFrame = document.getElementById('map-frame');

  function loadMap(lat, lng) {
    const query = encodeURIComponent('cancer hospital oncology near me');
    const src   = `https://www.google.com/maps/embed/v1/search`
                + `?key=AIzaSyD-9tSrke72PouQMnMX-a7eZSW0jkFMBWY`
                + `&q=${query}`
                + `&center=${lat},${lng}`
                + `&zoom=13`;

    // Fallback — use plain search embed without key (works for most regions)
    const fallback = `https://maps.google.com/maps?q=cancer+hospital+oncology&ll=${lat},${lng}&z=13&output=embed`;

    mapFrame.src = fallback;
    statusEl.parentElement.style.background = '#f0faf7';
    statusEl.parentElement.style.borderColor = '#b8dbd6';
    statusEl.parentElement.querySelector('.dot').style.background = '#1a6b5e';
    statusEl.innerHTML = `<strong>Location detected</strong> &nbsp;·&nbsp; Showing nearby cancer hospitals`;
  }

  function handleError(err) {
    // If geolocation fails, fall back to generic search
    const fallback = `https://maps.google.com/maps?q=cancer+hospital+oncology&z=12&output=embed`;
    mapFrame.src   = fallback;
    statusEl.parentElement.querySelector('.dot').style.animation = 'none';
    statusEl.parentElement.querySelector('.dot').style.background = '#d68910';
    statusEl.innerHTML = `Location access denied &nbsp;·&nbsp; Showing general results. Enable location for nearby hospitals.`;
  }

  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => loadMap(pos.coords.latitude, pos.coords.longitude),
      handleError,
      { timeout: 8000 }
    );
  } else {
    handleError();
  }
</script>
</body>
</html>
"""

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

*, *::before, *::after { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #f5f7f6;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stMainBlockContainer"] {
    padding: 0 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8e6;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.5rem; }

.navbar {
    background: #ffffff;
    border-bottom: 1px solid #e2e8e6;
    padding: 1rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -2.5rem 2rem -2.5rem;
}
.navbar-brand  { display:flex; align-items:center; gap:0.6rem; }
.navbar-title  { font-family:'DM Serif Display',serif; font-size:1.6rem; color:#1a6b5e; margin:0; }
.navbar-sub    { font-size:0.78rem; color:#7a9e97; margin:0; font-weight:500; letter-spacing:0.03em; }
.navbar-badge  {
    background:#e8f4f2; color:#1a6b5e; border:1px solid #b8dbd6;
    padding:0.3rem 0.9rem; border-radius:20px; font-size:0.75rem; font-weight:600;
}

.stat-card {
    background:#ffffff; border-radius:12px; padding:1.1rem 1.4rem;
    border:1px solid #e2e8e6; box-shadow:0 1px 4px rgba(26,107,94,0.06);
}
.stat-val {
    font-size:1.6rem; font-weight:700; color:#1a3a36; margin:0;
    font-family:'DM Serif Display',serif;
    display:flex; align-items:center; gap:0.4rem; line-height:1.2;
}
.stat-lbl {
    font-size:0.72rem; color:#7a9e97; margin:0.3rem 0 0 0;
    text-transform:uppercase; letter-spacing:0.1em; font-weight:600;
    display:flex; align-items:center; gap:0.35rem;
}

.panel {
    background:#ffffff; border-radius:14px;
    border:1px solid #e2e8e6; overflow:hidden;
    box-shadow:0 2px 8px rgba(26,107,94,0.06);
}
.panel-header {
    padding:0.75rem 1.2rem; border-bottom:1px solid #e8f0ee;
    background:#fafcfb; display:flex; align-items:center; gap:0.5rem;
    font-size:0.72rem; font-weight:700; color:#4a7a73;
    letter-spacing:0.12em; text-transform:uppercase;
}
.panel-body { padding:1rem; }

.result-panel {
    background:#ffffff; border-radius:14px;
    border:1px solid #e2e8e6; box-shadow:0 2px 8px rgba(26,107,94,0.06);
    overflow:hidden;
}
.result-header {
    padding:0.75rem 1.2rem; border-bottom:1px solid #e8f0ee;
    background:#fafcfb; font-size:0.72rem; font-weight:700; color:#4a7a73;
    letter-spacing:0.12em; text-transform:uppercase;
    display:flex; align-items:center; gap:0.5rem;
}
.result-body { padding:1.2rem; }

.cancer-card {
    background:#fff5f5; border:1px solid #f5c6c6;
    border-left:4px solid #c0392b; border-radius:10px; padding:1rem 1.2rem;
}
.suspicious-card {
    background:#fffbf0; border:1px solid #f5e0a0;
    border-left:4px solid #d68910; border-radius:10px; padding:1rem 1.2rem;
}
.safe-card {
    background:#f0faf7; border:1px solid #b8dbd6;
    border-left:4px solid #1a6b5e; border-radius:10px; padding:1rem 1.2rem;
}
.card-title {
    font-size:1.25rem; font-weight:700; margin:0 0 0.3rem 0;
    display:flex; align-items:center; gap:0.5rem;
    font-family:'DM Serif Display',serif;
}
.card-conf  { font-size:0.88rem; color:#4a5568; margin:0 0 0.6rem 0; font-weight:500; }
.card-note  {
    font-size:0.82rem; line-height:1.6;
    display:flex; gap:0.5rem; align-items:flex-start; margin:0;
}

/* ── MAP BUTTON ── */
.map-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 0.9rem;
    background: #c0392b;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.1rem;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    letter-spacing: 0.03em;
    transition: background 0.2s;
    text-decoration: none;
    font-family: 'DM Sans', sans-serif;
}
.map-btn:hover { background: #a93226; }
.map-section-header {
    font-size:0.7rem; font-weight:700; color:#7a9e97;
    letter-spacing:0.15em; text-transform:uppercase;
    display:flex; align-items:center; gap:0.4rem;
    margin:1.5rem 0 0.8rem 0; padding-bottom:0.5rem;
    border-bottom:1px solid #e8f0ee;
}

.threshold-box {
    background:#f5f7f6; border:1px solid #e2e8e6; border-radius:8px;
    padding:0.7rem 1rem; font-size:0.75rem; color:#7a9e97;
    margin-top:1rem; display:flex; align-items:center; gap:0.5rem; line-height:1.6;
}
.legend {
    background:#f5f7f6; border:1px solid #e2e8e6; border-radius:8px;
    padding:0.6rem 1rem; font-size:0.78rem; color:#4a7a73;
    display:flex; gap:1.2rem; align-items:center; margin-top:0.5rem; flex-wrap:wrap;
}
.step-card {
    background:#ffffff; border:1px solid #e2e8e6; border-radius:12px;
    padding:1.4rem; box-shadow:0 1px 4px rgba(26,107,94,0.04);
}
.step-num {
    width:28px; height:28px; background:#e8f4f2; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.78rem; font-weight:700; color:#1a6b5e; margin-bottom:0.7rem;
}
.step-title { font-size:0.92rem; font-weight:700; color:#1a3a36; margin:0 0 0.3rem 0; }
.step-desc  { font-size:0.82rem; color:#7a9e97; margin:0; line-height:1.5; }

.sb-logo    { font-family:'DM Serif Display',serif; font-size:1.5rem; color:#1a6b5e; margin:0 0 0.1rem 0; }
.sb-team    { font-size:0.7rem; font-weight:700; color:#7a9e97; letter-spacing:0.12em; text-transform:uppercase; margin:0 0 0.8rem 0; }
.sb-member  { display:flex; align-items:center; gap:0.5rem; font-size:0.82rem; color:#4a7a73; padding:0.3rem 0; }
.sb-divider { border:none; border-top:1px solid #e8f0ee; margin:0.8rem 0; }
.sb-section { font-size:0.68rem; font-weight:700; color:#7a9e97; letter-spacing:0.12em; text-transform:uppercase; margin:0.8rem 0 0.5rem 0; }
.sb-row     { display:flex; align-items:flex-start; gap:0.5rem; padding:0.25rem 0; font-size:0.82rem; }
.sb-key     { color:#4a7a73; font-weight:600; min-width:90px; }
.sb-val     { color:#1a3a36; }

.disclaimer {
    background:#f5f7f6; border:1px solid #e2e8e6; border-radius:8px;
    padding:0.8rem 1.2rem; font-size:0.78rem; color:#7a9e97;
    display:flex; align-items:flex-start; gap:0.5rem; line-height:1.6; margin-top:1.5rem;
}

[data-testid="stFileUploader"] {
    background:#ffffff; border:2px dashed #b8dbd6; border-radius:12px;
}
.stProgress > div > div {
    background:linear-gradient(90deg,#1a6b5e,#2ecc71) !important;
    border-radius:4px !important;
}
[data-testid="stProgressBar"] { background:#e8f0ee !important; border-radius:4px !important; }
.stAlert                         { display:none !important; }
[data-testid="stNotification"]   { display:none !important; }
div[data-baseweb="notification"] { display:none !important; }
hr { border-color:#e2e8e6 !important; margin:1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Navbar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="navbar">
    <div class="navbar-brand">
        {icon("microscope")}
        <div>
            <p class="navbar-title">ONCOAi</p>
            <p class="navbar-sub">ORAL CANCER DETECTION SYSTEM</p>
            <p class="navbar-badge">MobileNetV2 &nbsp;·&nbsp; 92.4% Accuracy</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<p style="font-size:0.82rem;color:#4a7a73;font-weight:600;
margin-bottom:0.4rem;display:flex;align-items:center;gap:0.4rem;">
    {icon("upload")} Upload oral cavity image (JPG / JPEG / PNG)
</p>""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload image", type=["jpg","jpeg","png"],
    label_visibility="collapsed"
)

# ── Session state for map toggle ──────────────────────────────────────────────
if 'show_map' not in st.session_state:
    st.session_state.show_map = False

# ══════════════════════════════════════════════════════════════════════════════
if uploaded is not None:
# ══════════════════════════════════════════════════════════════════════════════

    image = Image.open(uploaded)

    with st.spinner("Analysing image..."):
        pred_class, confidence, all_probs = predict(image)
        diagnosis, diag_prob = get_diagnosis(all_probs)
        pred_index     = 0 if pred_class == 'CANCER' else 1
        model_obj      = get_model()
        last_conv      = get_last_conv_layer_name(model_obj)
        gradcam_img, _ = generate_gradcam(model_obj, image, pred_index, last_conv)

    cancer_prob    = all_probs.get('CANCER', 0.0)
    noncancer_prob = all_probs.get('NON CANCER', 0.0)

    if diagnosis == 'CANCER':
        diag_color = '#c0392b'; diag_icon = icon("warning")
    elif diagnosis == 'SUSPICIOUS':
        diag_color = '#d68910'; diag_icon = icon("suspicious")
    else:
        diag_color = '#1a6b5e'; diag_icon = icon("check")

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-val">{confidence:.1f}%</p>
            <p class="stat-lbl">{icon("gauge")} Confidence Score</p>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-val" style="color:{diag_color};">{diag_icon} {diagnosis}</p>
            <p class="stat-lbl">{icon("stethoscope")} AI Diagnosis</p>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-val" style="color:#c0392b;">{cancer_prob:.1f}%</p>
            <p class="stat-lbl">{icon("biohazard")} Cancer Probability</p>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown(f"""
        <div class="stat-card">
            <p class="stat-val" style="color:#1a6b5e;">{noncancer_prob:.1f}%</p>
            <p class="stat-lbl">{icon("shield")} Non-Cancer Probability</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── 3 column layout ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.1, 1.1, 1])

    with col1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">{icon("upload")} Original Image</div>
            <div class="panel-body">""", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">{icon("fire")} Grad-CAM Heatmap</div>
            <div class="panel-body">""", unsafe_allow_html=True)
        st.image(gradcam_img, use_container_width=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="legend">
            <span style="display:flex;align-items:center;gap:0.4rem;">
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" fill="#c0392b"/></svg>
                Red / Yellow — Suspicious region
            </span>
            <span style="display:flex;align-items:center;gap:0.4rem;">
                <svg width="10" height="10"><circle cx="5" cy="5" r="5" fill="#2980b9"/></svg>
                Blue / Purple — Normal tissue
            </span>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="result-panel">
            <div class="result-header">{icon("chart")} Diagnosis Result</div>
            <div class="result-body">""", unsafe_allow_html=True)

        if diagnosis == 'CANCER':
            st.markdown(f"""
            <div class="cancer-card">
                <p class="card-title" style="color:#c0392b;">
                    {icon("warning")} Cancer Detected
                </p>
                <p class="card-conf">Confidence: <b>{cancer_prob:.1f}%</b></p>
                <p class="card-note" style="color:#7b241c;">
                    {icon("hospital")}
                    High probability of malignancy. Please consult an oncologist
                    or dental specialist immediately.
                </p>
            </div>""", unsafe_allow_html=True)

        elif diagnosis == 'SUSPICIOUS':
            st.markdown(f"""
            <div class="suspicious-card">
                <p class="card-title" style="color:#d68910;">
                    {icon("suspicious")} Suspicious Finding
                </p>
                <p class="card-conf">Cancer Probability: <b>{cancer_prob:.1f}%</b></p>
                <p class="card-note" style="color:#7d6608;">
                    {icon("hospital")}
                    Borderline result. Clinical evaluation is strongly recommended.
                </p>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="safe-card">
                <p class="card-title" style="color:#1a6b5e;">
                    {icon("check")} No Cancer Detected
                </p>
                <p class="card-conf">Confidence: <b>{noncancer_prob:.1f}%</b></p>
                <p class="card-note" style="color:#1a4a42;">
                    {icon("calendar")}
                    No malignant lesion detected. Continue regular dental checkups.
                </p>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="threshold-box">
            {icon("info")}
            <span><b>Thresholds:</b>&nbsp;
            Cancer ≥ 60% &nbsp;·&nbsp; Suspicious 30–60% &nbsp;·&nbsp; Non-Cancer &lt; 30%
            </span>
        </div>""", unsafe_allow_html=True)

        # ── MAP BUTTON — only show for CANCER or SUSPICIOUS ───────────────────
        if diagnosis in ['CANCER', 'SUSPICIOUS']:
            st.markdown(f"""
            <div style="margin-top:1rem; padding-top:1rem; border-top:1px solid #e8f0ee;">
                <p style="font-size:0.72rem;font-weight:700;color:#7a9e97;
                letter-spacing:0.1em;text-transform:uppercase;margin:0 0 0.5rem 0;">
                    {icon("map")} Find Nearby Hospitals
                </p>
                <p style="font-size:0.78rem;color:#4a7a73;margin:0 0 0.6rem 0;">
                    Click to locate cancer &amp; oncology hospitals near you
                </p>
            </div>""", unsafe_allow_html=True)

            if st.button(
                "📍  Show Nearby Cancer Hospitals",
                key="map_btn",
                type="primary",
                use_container_width=True
            ):
                st.session_state.show_map = not st.session_state.show_map

        st.markdown("</div></div>", unsafe_allow_html=True)

        # ── Probability bars ───────────────────────────────────────────────
        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <p style="font-size:0.7rem;font-weight:700;color:#7a9e97;letter-spacing:0.12em;
        text-transform:uppercase;margin:0 0 0.5rem 0;
        display:flex;align-items:center;gap:0.4rem;">
            {icon("chart")} Probability Breakdown
        </p>""", unsafe_allow_html=True)
        st.progress(float(cancer_prob / 100),    text=f"CANCER      {cancer_prob:.1f}%")
        st.progress(float(noncancer_prob / 100), text=f"NON CANCER  {noncancer_prob:.1f}%")

    # ── MAP SECTION ───────────────────────────────────────────────────────────
    if st.session_state.show_map and diagnosis in ['CANCER', 'SUSPICIOUS']:
        st.markdown(f"""
        <div class="map-section-header">
            {icon("location")} Nearby Cancer &amp; Oncology Hospitals
        </div>""", unsafe_allow_html=True)

        st.components.v1.html(MAP_HTML, height=520, scrolling=False)

        st.markdown(f"""
        <div class="disclaimer" style="margin-top:0.8rem;">
            {icon("info")}
            <span>
                <b>Note:</b> This map shows hospitals near your current location.
                Allow location access when prompted by your browser for accurate results.
                Always call ahead to confirm oncology services availability.
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="disclaimer">
        {icon("info")}
        <span>
            <b>Medical Disclaimer:</b> ONCOAi is an AI-assisted screening tool only.
            It is NOT a substitute for professional medical diagnosis. Always consult
            a qualified dentist or oncologist for evaluation and treatment.
        </span>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
else:
    st.session_state.show_map = False
# ══════════════════════════════════════════════════════════════════════════════

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <p style="font-size:0.7rem;font-weight:700;color:#7a9e97;
    letter-spacing:0.15em;text-transform:uppercase;margin:1.5rem 0 0.8rem 0;
    display:flex;align-items:center;gap:0.4rem;">
        {icon("brain")} How It Works
    </p>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1","upload",      "Upload",   "Upload a clear JPG or PNG photo of the oral cavity area"),
        ("2","brain",       "Analyse",  "MobileNetV2 deep learning model processes and extracts features"),
        ("3","stethoscope", "Diagnose", "AI classifies as Cancer, Suspicious, or Non-Cancer"),
        ("4","fire",        "Explain",  "Grad-CAM heatmap highlights the suspicious region of interest"),
    ]
    for col,(num,ico,title,desc) in zip([c1,c2,c3,c4],steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                {icon(ico)}
                <p class="step-title" style="margin-top:0.5rem;">{title}</p>
                <p class="step-desc">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <p style="font-size:0.7rem;font-weight:700;color:#7a9e97;
    letter-spacing:0.15em;text-transform:uppercase;margin:2rem 0 0.8rem 0;
    display:flex;align-items:center;gap:0.4rem;">
        {icon("chart")} Diagnosis Categories
    </p>""", unsafe_allow_html=True)

    t1,t2,t3 = st.columns(3)
    thresholds = [
        ("#c0392b","#fff5f5","#f5c6c6","warning",   "Cancer Detected",
         "Cancer probability ≥ 60%",
         "High likelihood of malignancy. Immediate specialist consultation required."),
        ("#d68910","#fffbf0","#f5e0a0","suspicious","Suspicious Finding",
         "Cancer probability 30% – 60%",
         "Borderline result. Clinical evaluation by a dentist strongly recommended."),
        ("#1a6b5e","#f0faf7","#b8dbd6","check",     "No Cancer Detected",
         "Cancer probability < 30%",
         "Low likelihood of malignancy. Continue routine dental checkups."),
    ]
    for col,(color,bg,border,ico,label,threshold,desc) in zip([t1,t2,t3],thresholds):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {border};
            border-left:4px solid {color};border-radius:12px;padding:1.2rem 1.4rem;">
                <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
                    {icon(ico)}
                    <span style="font-weight:700;color:{color};font-size:0.95rem;
                    font-family:'DM Serif Display',serif;">{label}</span>
                </div>
                <p style="font-size:0.78rem;color:#4a5568;margin:0 0 0.4rem 0;font-weight:600;">
                    {threshold}</p>
                <p style="font-size:0.78rem;color:#7a9e97;margin:0;line-height:1.5;">{desc}</p>
            </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem;">
        {icon("microscope")}
        <p class="sb-logo">ONCOAi</p>
    </div>
    <p class="sb-team">Team MediScope</p>
    <div class="sb-member">{icon("user")} Sasmita D &nbsp;— 727823TUCS305</div>
    <div class="sb-member">{icon("user")} Sedhupathi R — 727823TUCS308</div>
    <div class="sb-member">{icon("user")} Shabin George — 727823TUCS310</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <p class="sb-section">Mentor</p>
    <div class="sb-member">{icon("user")} Dr. Udhayamoorthi M</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown('<p class="sb-section">Model Details</p>', unsafe_allow_html=True)
    for ico,key,val in [
        ("brain",       "Architecture",  "MobileNetV2"),
        ("chart",       "Pre-trained",   "ImageNet"),
        ("upload",      "Dataset",       "1700 images"),
        ("gauge",       "Input Size",    "224 × 224 px"),
        ("chart",       "Accuracy",      "92.4% (F1 = 0.92)"),
        ("fire",        "Explainability","Grad-CAM"),
        ("stethoscope", "Output",        "3 Categories"),
    ]:
        st.markdown(f"""
        <div class="sb-row">{icon(ico)}
            <span class="sb-key">{key}</span>
            <span class="sb-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="sb-divider">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-row">{icon("hospital")}
        <span class="sb-val">Dept. of CSE</span>
    </div>""", unsafe_allow_html=True)

# ── Chatbot Integration ──────────────────────────────────────────────────────
initialize_chatbot_session()

# Add chatbot section with divider
st.markdown(f"""
<div style="margin-top:3rem;padding-top:2rem;border-top:1px solid #e2e8e6;">
</div>""", unsafe_allow_html=True)

st.markdown(f"""
<p style="font-size:0.7rem;font-weight:700;color:#7a9e97;
letter-spacing:0.15em;text-transform:uppercase;margin:0.8rem 0;
display:flex;align-items:center;gap:0.4rem;">
    💬 Need Help? Ask Our AI Assistant
</p>""", unsafe_allow_html=True)

render_quick_links()

st.markdown(f"""
<div style="margin-top:2rem;"></div>""", unsafe_allow_html=True)

with st.expander("💬 Chat with ONCOAi Assistant", expanded=False):
    render_simple_chat()
