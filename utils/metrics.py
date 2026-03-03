"""
metrics.py — Plotting helpers for training results
Owner: Sedhupathi R | ONCOAi Team MediScope
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training(history, save_path='reports/figures/training_curves.png'):
    """Plot and save accuracy + loss curves from Keras history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'],     color='#028090', label='Train')
    ax1.plot(history.history['val_accuracy'], color='#e74c3c',
             linestyle='--', label='Validation')
    ax1.set_title('Model Accuracy', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(history.history['loss'],     color='#028090', label='Train')
    ax2.plot(history.history['val_loss'], color='#e74c3c',
             linestyle='--', label='Validation')
    ax2.set_title('Model Loss', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.suptitle('ONCOAi — Training History', fontweight='bold', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes,
                          save_path='reports/figures/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5)
    plt.title('Confusion Matrix — ONCOAi', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {save_path}")