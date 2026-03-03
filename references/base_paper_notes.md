# Reference Notes

## Base Paper
**Title:** Automated Detection and Classification of Oral Lesions Using
Deep Learning for Early Detection of Oral Cancer
**Authors:** Welikala R.A. et al.
**Journal:** IEEE Access, Volume 8, 2020
**DOI:** 10.1109/ACCESS.2020.3010180
**Model used:** ResNet-101 + Faster R-CNN
**Best result:** F1 = 87.07%

## Reference Repo
**URL:** https://github.com/PrateekDutta2001/Oral-Cancer-Detection
**Models:** CNN from scratch, ResNet50 TL, VGG19 TL
**Dataset:** https://drive.google.com/drive/folders/1uL3LTG60HsMNkctTbbkNi9aulXmUMP5z

## How ONCOAi Improves on These
| Aspect | Reference Repo | ONCOAi |
|--------|----------------|--------|
| Model | ResNet50 / VGG19 | EfficientNetB0 |
| Explainability | None | Grad-CAM heatmap |
| Interface | None | Streamlit web app |
| Severity | Binary | 3-class |
| Mobile | No | Planned (TFLite) |
```

---

## ✅ Final Folder Structure in VS Code

After creating all files, your VS Code Explorer should look exactly like this:
```
OncoAi/
├── .gitignore
├── requirements.txt
├── README.md
│
├── dataset/
│   └── dataset_info.md          ← Sasmita fills this
│
├── notebooks/                   ← Colab notebooks go here
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_gradcam.ipynb
│
├── model/
│   ├── saved_models/
│   │   └── oral_cancer_detection.h5   ← Shabin downloads this
│   └── checkpoints/
│
├── app/
│   ├── app.py                   ← Shabin
│   └── predictor.py             ← Shabin
│
├── utils/
│   ├── preprocess.py            ← Sasmita
│   └── metrics.py               ← Sedhupathi
│
├── reports/
│   └── figures/                 ← Screenshots go here
│
└── references/
    └── base_paper_notes.md