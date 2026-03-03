"""
create_test_model.py
Builds EfficientNetB0 model and saves as .h5
Run ONCE from OncoAi root folder:
    python model/create_test_model.py
Owner: Shabin George | ONCOAi
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model
import os

print("=" * 50)
print("ONCOAi — Model Builder")
print("=" * 50)
print(f"TensorFlow: {tf.__version__}")
print("Building EfficientNetB0...")

# ── Build ─────────────────────────────────────────────────────────────────────
base = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False

inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)

# 2 classes → CANCER (index 0), NON CANCER (index 1)
outputs = layers.Dense(2, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ── Save ──────────────────────────────────────────────────────────────────────
save_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'saved_models',
    'oral_cancer_detection.h5'
)
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save(save_path)

size_mb = os.path.getsize(save_path) / 1024 / 1024
print(f"\n✅ Model saved!")
print(f"   Path: {save_path}")
print(f"   Size: {size_mb:.1f} MB")
print("\nModel Summary:")
model.summary()