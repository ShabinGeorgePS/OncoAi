import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE    = (224, 224)
RANDOM_SEED = 42


def load_and_resize(image_path):
    """Load a single image and resize to 224x224."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    return img


def split_dataset(raw_dir, processed_dir, train=0.70, val=0.15, test=0.15):
    """
    Split dataset from raw_dir into train/val/test under processed_dir.

    raw_dir structure:
        raw_dir/
            normal/   <- folder name must match class
            cancer/

    processed_dir output:
        processed_dir/
            train/normal/, train/cancer/
            val/normal/,   val/cancer/
            test/normal/,  test/cancer/
    """
    classes = [d for d in os.listdir(raw_dir)
               if os.path.isdir(os.path.join(raw_dir, d))]

    print(f"Found classes: {classes}")

    for cls in classes:
        src  = os.path.join(raw_dir, cls)
        imgs = [f for f in os.listdir(src)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        train_f, temp  = train_test_split(imgs, test_size=(1 - train),
                                          random_state=RANDOM_SEED)
        val_f, test_f  = train_test_split(temp, test_size=0.5,
                                          random_state=RANDOM_SEED)

        print(f"{cls}: train={len(train_f)}, val={len(val_f)}, test={len(test_f)}")

        for split, files in [('train', train_f), ('val', val_f), ('test', test_f)]:
            dest_dir = os.path.join(processed_dir, split, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in files:
                img = load_and_resize(os.path.join(src, fname))
                img.save(os.path.join(dest_dir, fname))

    print("Dataset split complete!")