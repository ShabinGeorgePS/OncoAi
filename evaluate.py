"""
evaluate.py — ONCOAi Model Accuracy Checker
Run with: python evaluate.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
WEIGHTS_PATH  = r'C:\Users\shabi\OncoAi\model\saved_models\oncoai_best.weights.h5'

# ✅ Update this to your dataset path
DATASET_PATH  = r"D:\OncoAi_datset\Oral Cancer"
CANCER_PATH   = os.path.join(DATASET_PATH, 'CANCER')
NONCANCER_PATH= os.path.join(DATASET_PATH, 'NON CANCER')

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
CLASS_NAMES   = ['CANCER', 'NON CANCER']

# ── Step 1 — Check dataset ────────────────────────────────────────────────────
print('=' * 55)
print('       ONCOAi — Model Evaluation')
print('=' * 55)

cancer_files    = [f for f in os.listdir(CANCER_PATH)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]
noncancer_files = [f for f in os.listdir(NONCANCER_PATH)
                   if f.lower().endswith(('.jpg','.jpeg','.png'))]

print(f'\nDataset found:')
print(f'  CANCER     : {len(cancer_files)} images')
print(f'  NON CANCER : {len(noncancer_files)} images')
print(f'  TOTAL      : {len(cancer_files) + len(noncancer_files)} images')

# ── Step 2 — Build model ──────────────────────────────────────────────────────
print('\nBuilding model...')
tf.keras.backend.clear_session()

inputs     = tf.keras.Input(shape=(224, 224, 3))
base_model = MobileNetV2(weights=None, include_top=False, input_tensor=inputs)
x          = base_model.output
x          = layers.GlobalAveragePooling2D()(x)
x          = layers.BatchNormalization()(x)
x          = layers.Dense(128, activation='relu')(x)
x          = layers.Dropout(0.3)(x)
x          = layers.Dense(64, activation='relu')(x)
x          = layers.Dropout(0.2)(x)
outputs    = layers.Dense(2, activation='softmax')(x)
model      = tf.keras.Model(inputs=inputs, outputs=outputs)

# ── Step 3 — Load weights ─────────────────────────────────────────────────────
print('Loading trained weights...')
model.load_weights(WEIGHTS_PATH)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print('✅ Model ready!')

# ── Step 4 — Create data generator ───────────────────────────────────────────
print('\nLoading dataset...')
datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f'  Class indices: {generator.class_indices}')
print(f'  Total samples: {generator.samples}')

# ── Step 5 — Evaluate ─────────────────────────────────────────────────────────
print('\nEvaluating model...')
loss, accuracy = model.evaluate(generator, verbose=1)

print(f'\n{"=" * 55}')
print(f'  Test Accuracy : {accuracy * 100:.2f}%')
print(f'  Test Loss     : {loss:.4f}')
print(f'{"=" * 55}')

# ── Step 6 — Classification report ───────────────────────────────────────────
print('\nGenerating predictions...')
y_pred = np.argmax(model.predict(generator, verbose=1), axis=1)
y_true = generator.classes
labels = list(generator.class_indices.keys())

print('\n--- Classification Report ---')
print(classification_report(y_true, y_pred, target_names=labels))

# ── Step 7 — Confusion matrix ─────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=labels,
    yticklabels=labels,
    linewidths=1,
    linecolor='gray'
)
ax.set_title('ONCOAi — Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
plt.tight_layout()

# Save confusion matrix
SAVE_PATH = r'C:\Users\shabi\OncoAi\reports\figures\confusion_matrix_local.png'
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
plt.savefig(SAVE_PATH, dpi=150)
plt.show()
print(f'\n✅ Confusion matrix saved to: {SAVE_PATH}')

# ── Step 8 — Per class summary ────────────────────────────────────────────────
print(f'\n--- Per Class Summary ---')
for i, label in enumerate(labels):
    correct   = cm[i][i]
    total     = cm[i].sum()
    class_acc = correct / total * 100
    print(f'  {label:12s} → {correct:3d}/{total:3d} correct ({class_acc:.1f}%)')

print(f'\n✅ Evaluation complete!')
