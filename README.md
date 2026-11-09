# DeepFER: Facial Emotion Recognition Using Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%202.x%20%7C%20Keras%203-orange.svg)](https://keras.io/)
[![Computer Vision](https://img.shields.io/badge/Vision-OpenCV%20%7C%20Pillow-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Executive Summary

**DeepFER (Deep Facial Emotion Recognition)** is an end-to-end deep learning and affective computing system engineered to detect and classify human facial expressions into seven discrete emotional states: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

In conventional human-computer interfaces, software remains emotionally blind—responding only to transactional keystrokes and clicks. DeepFER bridges this gap by delivering automated affective intelligence capable of empowering empathetic customer support workflows, real-time driver drowsiness/fatigue monitoring, interactive educational personalization, and clinical mental health diagnostics.

The core deliverable of this project is a comprehensive, production-ready Jupyter notebook:  
[`DeepFER_Facial_Emotion_Recognition.ipynb`](DeepFER_Facial_Emotion_Recognition.ipynb)

---

## 🔬 Dataset Overview & Data Hygiene

The underlying dataset comprises **35,887 raw grayscale images** ($48 \times 48$ resolution) organized into disk directories by emotion category across training and validation splits.

```
data/images/
├── train/
│   ├── angry/       (3,993 images)
│   ├── disgust/       (436 images)
│   ├── fear/        (4,103 images)
│   ├── happy/       (7,164 images)
│   ├── neutral/     (4,982 images)
│   ├── sad/         (4,938 images)
│   └── surprise/    (3,205 images)  [Total: 28,821]
└── validation/
    ├── angry/         (960 images)
    ├── disgust/       (111 images)
    ├── fear/        (1,018 images)
    ├── happy/       (1,825 images)
    ├── neutral/     (1,216 images)
    ├── sad/         (1,139 images)
    └── surprise/      (797 images)  [Total: 7,066]
```

### Critical Data Challenges Addressed
1. **Severe Class Disparity (20:1 Imbalance)**:
   - `happy` constitutes over 25.8% of the data, while `disgust` accounts for only ~1.3% (547 total images).
   - Evaluating models using standard overall accuracy is actively misleading because a trivial model predicting majority classes can show high accuracy while failing completely on minority distress classes.
   - **Solution**: Balanced class-weighted cross-entropy loss and **Macro-Averaged F1-Score** as the primary champion selection metric.
2. **Data Leakage & Duplicate Removal**:
   - SHA-256 pixel content hashing revealed **1,853 duplicate image instances**, including **510 identical images present in both training and validation sets** (cross-split leakage) and **57 unique images assigned conflicting labels** across different classes.
   - **Solution**: Cleaned manifest removes contradictory labels and deduplicates cross-split items, retaining **33,977 verified unique images**.
3. **Rigorous Experimental Partitioning**:
   - A dedicated **held-out stratified test set (15%, 4,052 images)** is carved out from the cleaned training pool and completely isolated from model selection and hyperparameter tuning.

---

## 🏗 Pipeline Architecture & Notebook Walkthrough

The notebook follows a structured, enterprise-grade machine learning lifecycle:

```
[Raw Image Directories]
          │
          ▼
1. Know Your Data ──────────► SHA-256 Hash Audit, Dimension/Mode Verification, Null Audit
          │
          ▼
2. Variable Understanding ──► Attribute Specifications & Unique Value Diagnostics
          │
          ▼
3. Data Wrangling ──────────► Conflict Purge, Cross-Split Leakage Elimination, Label Encoding
          │
          ▼
4. EDA & Visualization ─────► 6 Strategic Charts (Frequencies, Gallery, Brightness, Mean Faces, Heatmap)
          │
          ▼
5. Hypothesis Testing ──────► Statistical Validation (Welch's t-tests, Mann-Whitney U, Chi-Square)
          │
          ▼
6. Feature Engineering ─────► In-Memory Tensor Ingestion, Class Weighting, Spatial Augmentation
          │
          ▼
7. Modeling & Evaluation ───► Baseline vs. Custom Deep CNN vs. MobileNetV2 (Stage 1 & 2)
          │
          ▼
8. Persistence & Inference ─► Model Serialization, Parity Check, Latency Benchmark (FPS)
          │
          ▼
9. Real-Time Webcam Demo ───► Live OpenCV Haar-Cascade Face ROI Extraction & Emotion Overlay
```

---

## 🧠 Evaluated Model Architectures

### 1. Majority Class Baseline
- Acts as the theoretical performance floor by consistently predicting the most frequent training category (`happy`).
- Demonstrates why macro-averaged metrics are essential (achieves near-zero Macro Precision, Recall, and F1 despite ~25% accuracy).

### 2. Custom Deep Convolutional Neural Network (CNN)
Tailored specifically for $48 \times 48 \times 1$ facial crops:
- **Input Augmentation & Rescaling**: In-graph random horizontal flips, subtle rotations ($\pm 5\%$), zooms ($\pm 10\%$), and $[0, 255] \to [0, 1]$ rescaling.
- **Convolutional Feature Backbone**:
  - *Block 1*: $2\times$ `Conv2D(32, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)` + `Dropout(0.25)`.
  - *Block 2*: $2\times$ `Conv2D(64, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)` + `Dropout(0.25)`.
  - *Block 3*: `Conv2D(128, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)` + `Dropout(0.30)`.
- **Classification Head**: `GlobalAveragePooling2D` $\to$ `Dense(256)` + `BatchNorm` + `ReLU` + `Dropout(0.50)` $\to$ `Dense(7, softmax)`.
- **Loss & Optimization**: Balanced class weights with Adam optimizer ($\eta = 10^{-3}$) and early stopping.

### 3. Transfer Learning & Fine-Tuning (MobileNetV2)
Leverages ImageNet visual feature detectors:
- **Input Adaptation**: Channel duplication $(48, 48, 1) \to (48, 48, 3)$ followed by bilinear resizing to $(96, 96, 3)$.
- **Two-Stage Training Strategy**:
  - *Stage 1 (Feature Extraction)*: Frozen MobileNetV2 backbone; only the custom classification head is trained ($\eta = 10^{-3}$).
  - *Stage 2 (Targeted Fine-Tuning)*: Top 20 convolutional layers unfrozen (with Batch Normalization layers locked to preserve running statistics) trained with a gentle learning rate ($\eta = 10^{-5}$).
  - *Safety Guardrail*: Compares validation Macro F1 across stages; restores Stage 1 weights if fine-tuning exhibits overfit.

---

## 📊 Evaluation & Production Artifacts

The champion model selected via validation leaderboard is thoroughly audited on the held-out test set:
- **Full Classification Report**: Per-class precision, recall, and F1-score across all 7 emotions.
- **Normalized Confusion Matrix**: Identifies cross-class confusion vectors (e.g., subtle differentiation between `fear` vs `surprise` or `sad` vs `neutral`).
- **Qualitative Error Audit**: Visual inspection of top misclassifications showing true vs. predicted emotion and prediction confidence.

### Serialized Deployment Artifacts
All production artifacts are exported to the `artifacts/` directory:
- `artifacts/deepfer.keras`: Standalone serialized model with embedded preprocessing and weights.
- `artifacts/metadata.json`: Deployment contract defining class mappings, input shapes, normalization expectations, and test metrics.
- `artifacts/validation_results.csv`: Comprehensive validation leaderboard across all evaluated paradigms.

---

## ⚡ Inference Latency & Real-Time Webcam Demo

The notebook includes production inference utilities:
- **`predict_face(image_path)`**: Standalone function preprocessing an input image, running inference, and returning a ranked probability distribution across all 7 emotions.
- **Hardware Latency Benchmark**: 50 warm-up runs calculating mean latency, P95 latency, and theoretical throughput (FPS). Real-time edge performance (>30 FPS) is verified on standard hardware.
- **`run_webcam(camera_index=0)`**: Full interactive OpenCV demo that:
  1. Captures live video feed.
  2. Detects frontal faces using Haar Feature-based Cascade Classifiers.
  3. Crops and resizes face regions of interest (ROI) to $48 \times 48$.
  4. Runs forward-pass emotion classification.
  5. Renders color-coded bounding boxes and real-time confidence overlays.

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Requirements
Ensure Python 3.11 or 3.12 is installed. Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Dependency Installation
Install the project dependencies:

```bash
pip install --upgrade pip
pip install tensorflow keras numpy pandas scipy scikit-learn pillow matplotlib opencv-python ipykernel
```

### 3. Running the Notebook
Launch JupyterLab or PyCharm / VS Code:

```bash
jupyter lab DeepFER_Facial_Emotion_Recognition.ipynb
```

Execute cells sequentially from top to bottom. The notebook will automatically:
1. Verify directory structure and discover image files.
2. Complete data wrangling and hash audits.
3. Render EDA plots and hypothesis tests.
4. Ingest tensors, train models with class weighting, and record metrics.
5. Serialize the champion model and evaluate on the unseen test set.

### 4. Running the Real-Time Webcam Interface
To test live emotion recognition via your computer's webcam, uncomment and call the function in the final cell:

```python
run_webcam(camera_index=0)
```
*Note: Press `q` or `ESC` while focused on the video window to cleanly release the webcam stream.*

---

## 📁 Repository Directory Structure

```
facial-emotion-recognition-deep-learning/
│
├── DeepFER_Facial_Emotion_Recognition.ipynb   # Main end-to-end deep learning notebook
├── README.md                                  # Project documentation & execution guide
│
├── data/
│   └── images/
│       ├── train/                             # Training images (7 emotion subdirectories)
│       └── validation/                        # Validation images (7 emotion subdirectories)
│
└── artifacts/                                 # Generated production model artifacts
    ├── deepfer.keras                          # Serialized champion model
    ├── metadata.json                          # Model contract & class mappings
    └── validation_results.csv                 # Model benchmark comparison table
```

---

## 👥 Contributors & Contact

- **Author**: Sai Abhiram Simhadri
- **Project**: Facial Emotion Recognition Using Deep Learning
- **Repository**: [https://github.com/saiabhiram-simhadri/facial-emotion-recognition-deep-learning](https://github.com/saiabhiram-simhadri/facial-emotion-recognition-deep-learning)
