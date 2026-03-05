
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
CLASS_NAMES = ["CANCER", "NON CANCER"]

def get_last_conv_layer(base_model):
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def generate_gradcam(model, pil_image, pred_index, last_conv_layer_name):
    """
    Generates Grad-CAM heatmap overlay on image.
    
    Args:
        model: trained Keras model
        pil_image: PIL Image object
        pred_index: predicted class index (0=CANCER, 1=NON CANCER)
        last_conv_layer_name: last conv layer name in base model
    
    Returns:
        overlay_pil: PIL Image with heatmap overlay
        heatmap: raw heatmap numpy array
    """
    # Preprocess
    img  = pil_image.convert("RGB").resize(IMG_SIZE)
    arr  = np.array(img, dtype=np.float32)
    arr  = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr  = np.expand_dims(arr, axis=0)

    # Grad-CAM
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(arr)
        class_channel = predictions[:, pred_index]

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap      = heatmap.numpy()

    # Overlay
    img_cv       = np.array(img)
    img_cv       = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    hmap_resized = cv2.resize(heatmap, IMG_SIZE)
    hmap_colored = np.uint8(255 * hmap_resized)
    hmap_colored = cv2.applyColorMap(hmap_colored, cv2.COLORMAP_JET)
    superimposed  = cv2.addWeighted(img_cv, 0.6, hmap_colored, 0.4, 0)
    superimposed  = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    overlay_pil   = Image.fromarray(superimposed)

    return overlay_pil, heatmap
