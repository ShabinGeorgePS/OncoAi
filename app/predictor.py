"""
predictor.py
Loads model and predicts on a single image.
Owner: Shabin George | ONCOAi Team MediScope
Dataset classes: CANCER, NON CANCER
"""

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = (224, 224)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    '../model/saved_models/oral_cancer_detection.h5'
)


CLASS_NAMES = ['CANCER', 'NON CANCER']

# ── Load model once ───────────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            print("Loading model...")
            _model = load_model(MODEL_PATH)
            print("Model loaded!")
        else:
            print(f"Model not found at: {MODEL_PATH}")
    return _model

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(pil_image):
    """
    Input : PIL Image
    Output: (class_name, confidence_percent, all_probabilities)
    """
    model = get_model()
    if model is None:
        return "Model not found", 0.0, [0.0, 0.0]

    # Preprocess
    img = pil_image.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)        # shape → (1, 224, 224, 3)

    # Predict
    probs      = model.predict(arr, verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    return pred_class, confidence, probs.tolist()