import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers

WEIGHTS_PATH = r'C:\Users\shabi\OncoAi\model\saved_models\oncoai_best.weights.h5'
IMG_SIZE     = (224, 224)
CLASS_NAMES  = ['CANCER', 'NON CANCER']

_model = None

def build_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    base   = MobileNetV2(weights=None, include_top=False, input_tensor=inputs)
    x      = base.output
    x      = layers.GlobalAveragePooling2D()(x)
    x      = layers.BatchNormalization()(x)
    x      = layers.Dense(128, activation='relu')(x)
    x      = layers.Dropout(0.3)(x)
    x      = layers.Dense(64, activation='relu')(x)
    x      = layers.Dropout(0.2)(x)
    outputs = layers.Dense(2, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def get_model():
    global _model
    if _model is None:
        print("Building model...")
        _model = build_model()
        print("Loading weights...")
        _model.load_weights(WEIGHTS_PATH)
        print("✅ Model ready!")
    return _model

def preprocess_image(image: Image.Image) -> np.ndarray:
    image    = image.convert('RGB').resize(IMG_SIZE)
    arr      = np.array(image, dtype=np.float32)
    arr      = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(image: Image.Image):
    model        = get_model()
    img_array    = preprocess_image(image)
    predictions  = model.predict(img_array)
    pred_index   = np.argmax(predictions[0])
    confidence   = float(predictions[0][pred_index]) * 100
    pred_class   = CLASS_NAMES[pred_index]
    all_probs    = {CLASS_NAMES[i]: float(predictions[0][i]) * 100 for i in range(2)}
    return pred_class, confidence, all_probs