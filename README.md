# DeepFER: Facial Emotion Recognition Using Deep Learning

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Framework-TensorFlow%202.x%20%7C%20Keras%203-orange.svg)](https://keras.io/)
[![Computer Vision](https://img.shields.io/badge/Vision-OpenCV%20%7C%20Pillow-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Project Overview

I developed **DeepFER (Deep Facial Emotion Recognition)** as an end-to-end computer vision project to classify human facial expressions into seven emotion categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Neutral**, **Sad**, and **Surprise**.

Most software applications treat users as emotionally neutral, responding only to clicks and keypresses. By introducing automated facial expression analysis, systems can detect driver drowsiness, alert customer support teams to caller frustration in real time, and help educators gauge student engagement during online sessions.

### Key Deliverables
- **Main Jupyter Notebook**: [`DeepFER_Facial_Emotion_Recognition.ipynb`](DeepFER_Facial_Emotion_Recognition.ipynb) (contains my complete data pipeline, visual analysis, hypothesis tests, model training, and webcam inference code)
- **Project Report (.docx)**: [`DeepFER_Project_Report.docx`](DeepFER_Project_Report.docx) (formal project report documenting the entire methodology, metrics, and error analysis)

---

## Dataset Architecture & Data Hygiene

I worked with a raw dataset of **35,887 grayscale images** ($48 \times 48$ resolution) organized into disk folders across seven emotion classes:

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

### Data Problems I Identified & Solved
1. **Severe 20:1 Class Imbalance**:
   - `happy` accounts for 25.8% of the data (8,989 images), while `disgust` accounts for only 1.5% (547 images).
   - If I only measured overall accuracy, a simple baseline guessing majority classes would get ~26% accuracy while scoring 0% recall on high-stakes distress emotions.
   - **My Solution**: I computed inverse class weights to penalize minority misclassifications during training and adopted **Macro-Averaged F1-Score** as my champion selection metric.
2. **Cryptographic SHA-256 Duplicate & Leakage Audit**:
   - Rather than trusting the raw folder splits, I computed SHA-256 hashes of the raw pixel buffers of all 35,887 images.
   - I found **1,853 duplicate image files**, including **510 identical images present in both train and validation splits** (cross-split leakage) and **57 unique images assigned conflicting labels** (e.g., the same face labeled `angry` in one folder and `sad` in another).
   - **My Solution**: I purged all 57 conflicting label groups and deduplicated cross-split items, leaving **33,977 verified clean images**.
3. **Dedicated Held-Out Test Vault**:
   - I carved out a **15% stratified test split (4,052 images)** from the cleaned training pool and locked it away completely until final model auditing.

---

## Engineering Pipeline

I structured the project into nine sequential stages:

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
6. Feature Engineering ─────► In-Memory uint8 Tensor Ingestion, Class Weighting, Spatial Augmentation
          │
          ▼
7. Modeling & Evaluation ───► Baseline vs. Custom Deep CNN vs. MobileNetV2 (Stage 1 & 2)
          │
          ▼
8. Persistence & Inference ─► Model Serialization, Parity Check, Latency Benchmark
          │
          ▼
9. Real-Time Webcam Demo ───► Live OpenCV Haar-Cascade Face ROI Extraction & Emotion Overlay
```

---

## Models I Implemented

### 1. Majority Class Baseline
- Consistently predicts the most frequent training category (`happy`).
- Demonstrates why macro metrics matter: achieves 26.13% accuracy, but near-zero Macro Precision (3.73%) and Macro F1 (5.92%).

### 2. Custom Deep CNN (Champion Architecture)
Designed specifically for $48 \times 48 \times 1$ inputs:
- **In-Graph Preprocessing**: Random horizontal flips, rotations ($\pm 5\%$), zooms ($\pm 10\%$), and embedded $[0, 255] \to [0, 1]$ rescaling.
- **Backbone**:
  - *Block 1*: $2\times$ `Conv2D(32, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)` + `SpatialDropout(0.25)`.
  - *Block 2*: $2\times$ `Conv2D(64, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)` + `SpatialDropout(0.25)`.
  - *Block 3 & 4*: `Conv2D(128, 3x3)` + `BatchNorm` + `ReLU` + `MaxPool(2x2)` + `SpatialDropout(0.30)`.
- **Classification Head**: `GlobalAveragePooling2D` $\to$ `Dense(256)` + `BatchNorm` + `ReLU` + `Dropout(0.50)` $\to$ `Dense(7, softmax)`. Using Global Average Pooling instead of `Flatten()` prevented dense parameter explosion and reduced overfitting.
- **Optimization**: Trained across full-dataset passes with Adam ($\eta = 10^{-3}$), dynamic learning rate reduction (`ReduceLROnPlateau`), early stopping, and balanced class weights.

### 3. Transfer Learning (MobileNetV2)
Leverages pre-trained ImageNet feature extractors:
- **Input Adapter**: In-graph channel replication $(48, 48, 1) \to (48, 48, 3)$ and bilinear resize to $(96, 96, 3)$.
- **Two-Phase Strategy**:
  - *Stage 1 (Feature Extraction)*: Frozen MobileNetV2 backbone (8 epochs); custom classification head trained with Adam ($\eta = 10^{-3}$).
  - *Stage 2 (Fine-Tuning)*: Top 40 convolutional layers unfrozen with a lower learning rate ($\eta = 10^{-5}$) and plateau decay.
  - *Engineering Guardrail*: Kept all `BatchNormalization` layers permanently set to `training=False` during fine-tuning to prevent small-batch sample noise from corrupting pre-trained ImageNet statistics.

---

## Validation Leaderboard & Test Results

### Validation Benchmark Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Result |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Custom Deep CNN** | **43.82%** | **40.10%** | **43.45%** | **39.30%** | **Champion Model** |
| MobileNetV2 Transfer | 35.76% | 32.37% | 32.65% | 31.16% | Runner-up |
| Majority Class Baseline | 26.13% | 3.73% | 14.29% | 5.92% | Baseline floor |

Custom Deep CNN decisively claimed champion status by delivering the highest balanced Macro F1 (39.30%) and overall accuracy (43.82%), outperforming MobileNetV2 on this $48 \times 48$ single-channel task.

### Final Held-Out Test Evaluation (4,052 Images)

| Emotion Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Angry** | 0.2814 | 0.3640 | 0.3174 | 566 |
| **Disgust** | 0.0799 | 0.5094 | 0.1381 | 53 |
| **Fear** | 0.2551 | 0.2175 | 0.2348 | 570 |
| **Happy** | 0.7051 | 0.6310 | 0.6660 | 1,046 |
| **Neutral** | 0.4604 | 0.3760 | 0.4139 | 726 |
| **Sad** | 0.4240 | 0.3106 | 0.3585 | 718 |
| **Surprise** | 0.5125 | 0.6059 | 0.5553 | 373 |
| **Overall / Macro Avg** | **0.3883** | **0.4306** | **0.3834** | **4,052** |
| **Overall Accuracy** | — | — | **0.4292** | **4,052** |

### Key Confusion Matrix Insights
- **Happy & Surprise Dominance**: The model achieved its highest precision and recall on `Happy` (70.51% precision, 63.10% recall) and `Surprise` (60.59% recall), where distinct facial cues (raised mouth corners, dropped jaws) provide unambiguous signals.
- **Disgust Minority Recall**: Balanced class weighting enabled the model to capture **50.94% recall** on the extreme minority class (`Disgust`, 53 test images), demonstrating that the loss penalty effectively prevented the network from ignoring rare classes.
- **Fear vs. Surprise**: Both expressions share raised eyebrows and open mouths, which are occasionally confused in $48 \times 48$ grayscale crops.
- **Sad vs. Neutral**: Subtle downward lip curvature is occasionally misclassified as relaxed neutral expressions in low-resolution crops without video motion dynamics.
- **Misclassification Reduction**: Total misclassifications were reduced to **2,313 / 4,052 (57.08%)**, a dramatic improvement over the initial baseline models.

---

## Production Artifacts & Deployment

All production files are exported to `artifacts/`:
- `artifacts/deepfer.keras`: Standalone serialized champion model with embedded preprocessing layers and trained weights (**2.12 MB**).
- `artifacts/metadata.json`: Deployment contract defining class labels, input dimensions, normalization range, and test scores.
- `artifacts/validation_results.csv`: Logged benchmark scores across all tested models.

### CPU Latency Benchmark
- **Mean Forward-Pass Latency**: **29.51 ms** per frame
- **95th Percentile (P95) Latency**: **38.86 ms**
- **Theoretical Throughput**: **33.9 FPS** (fully edge capable for real-time video streams > 30 FPS)

### Real-Time OpenCV Webcam Demo
In the notebook, I built `run_webcam(camera_index=0)`:
1. Captures live webcam frames using OpenCV.
2. Detects frontal faces via Haar Feature Cascade (`haarcascade_frontalface_default.xml`).
3. Crops and resizes face regions to $48 \times 48$.
4. Runs forward-pass emotion classification.
5. Displays color-coded bounding boxes and confidence scores over the live video stream.

---

## Setup & Quickstart

### 1. Environment Setup
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Requirements
```bash
pip install --upgrade pip
pip install tensorflow keras numpy pandas scipy scikit-learn pillow matplotlib opencv-python ipykernel python-docx
```

### 3. Run the Project
```bash
jupyter lab DeepFER_Facial_Emotion_Recognition.ipynb
```
Run the cells sequentially to reproduce the data cleaning, exploratory charts, statistical tests, model training, and test evaluations.

---

## Repository Structure

```
facial-emotion-recognition-deep-learning/
│
├── DeepFER_Facial_Emotion_Recognition.ipynb   # Main end-to-end deep learning notebook
├── DeepFER_Project_Report.docx                # Formal project report (Word document)
├── README.md                                  # Project overview and execution guide
│
├── data/
│   └── images/
│       ├── train/                             # Cleaned training images (7 classes)
│       └── validation/                        # Cleaned validation images (7 classes)
│
├── artifacts/                                 # Serialized production artifacts
│   ├── deepfer.keras                          # Trained champion model (2.12 MB)
│   ├── metadata.json                          # Model contract & class mappings
│   └── validation_results.csv                 # Model benchmark comparison table
│
└── docs/                                      # Project documentation & figures
    ├── DeepFER_Project_Report.docx            # Documentation copy of project report
    └── report_figures/                        # Generated evaluation charts & visual plots
```

---

## Author

- **Sai Abhiram Simhadri**
- **Project**: DeepFER — Facial Emotion Recognition Using Deep Learning
- **Repository**: [https://github.com/saiabhiram-simhadri/facial-emotion-recognition-deep-learning](https://github.com/saiabhiram-simhadri/facial-emotion-recognition-deep-learning)
