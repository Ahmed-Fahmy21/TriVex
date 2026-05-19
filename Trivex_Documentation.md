# Trivex — Multimodal Behavioral Analysis System

> An AI system that analyzes human behavior in videos through **4 parallel pipelines** (Body, Face, Voice, Text) and produces **4 behavioral judgments**: Arousal & Valence, Interview Assessment, Emotional State, and Engagement Level.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Pipelines](#3-pipelines)
4. [Fusion Layer](#4-fusion-layer)
5. [Judgment Heads](#5-judgment-heads)
6. [Training Strategy](#6-training-strategy)
7. [Datasets](#7-datasets)
8. [Video Processing](#8-video-processing)
9. [Output Format](#9-output-format)
10. [Installation](#10-installation)
11. [Usage](#11-usage)
12. [Project Structure](#12-project-structure)
13. [Technical Requirements](#13-technical-requirements)
14. [Development Roadmap](#14-development-roadmap)

---

## 1. Overview

Trivex is a multimodal deep learning system that analyzes pre-recorded videos. Each video is split into **5-second windows**, processed through four parallel pretrained pipelines, fused together, and evaluated by four specialized judgment heads.

**Core Principles**

| Principle | Description |
|---|---|
| Transfer Learning | Use free pretrained models; train only small custom layers |
| Independent Models | Each judgment task is a separate, fully-trained model |
| Sliding Window | Process long videos in overlapping chunks |
| Tiered Dimensions | Each modality gets a dimension size matching its richness |

**Key Design Decision — 4 Independent Models**

Instead of one multi-task model with a shared backbone, Trivex uses **4 fully independent models**, one per judgment task. This means:

- ✅ Each model is trained only on its relevant dataset
- ✅ No negative transfer between tasks
- ✅ Easier to debug and develop incrementally
- ✅ No need to align labels across datasets

---

## 2. Architecture

### Per-Model Architecture

```
INPUT: Video + Audio
         │
  ┌──────┼──────┬──────┐
  ▼      ▼      ▼      ▼
BODY   FACE  VOICE   TEXT
 256    384    512    512
  └──────┴──────┴──────┘
             │
        FUSION LAYER
       Concat + MLP
             │ 512
             ▼
           HEAD
      (different per model)
             │
             ▼
        PREDICTION
```

### Component Summary

| Component | Type | Output Dim |
|---|---|---|
| Pipeline 1 — Body | MediaPipe Pose + Encoder | 256 |
| Pipeline 2 — Face | Face Mesh + DeepFace + Encoder | 384 |
| Pipeline 3 — Voice | Wav2Vec 2.0 (mean+std → 1536) + Classical (166) + Encoder | 512 |
| Pipeline 4 — Text | Whisper + BERT + Encoder | 512 |
| Fusion Layer | Concatenation + MLP | 512 |
| Head 1 — Arousal/Valence | Small MLP (regression, Tanh) | 2 continuous |
| Head 2 — Interview | Small MLP (regression, linear) | 5 continuous (Big Five) |
| Head 3 — Emotion | Small MLP (Softmax) | 7 classes |
| Head 4 — Engagement | Small MLP (Softmax) | 4 classes |

### Why Tiered Dimensions?

Each modality has a different natural richness. Squeezing everything into 128 dims creates a bottleneck; tiered dims match capacity to content.

| Pipeline | Raw Feature Size | Output Dim | Reasoning |
|---|---|---|---|
| Body | 132 / frame | 256 | Keypoints are simple — 256 is sufficient |
| Face | 1916 / frame | 384 | Landmarks + appearance need more capacity |
| Voice | 1702 / clip (1536 w2v + 166 classical) | 512 | Wav2Vec mean+std gives rich prosody; classical adds interpretable handcrafted cues |
| Text | 768 | 512 | BERT semantics need preservation |

---

## 3. Pipelines

> All pretrained models are **frozen**. Only the small custom encoders on top are trained.

### 3.1 Pipeline 1 — Body

**Pretrained model:** MediaPipe Pose (Google)  
**Input:** Video frames  
**Raw output:** 33 keypoints × 4 features = **132 per frame**

```
MediaPipe features (132/frame, ~150 frames)
  → Linear(132 → 256)
  → LSTM (1 layer, 256 hidden)
  → Last hidden state
  → Linear(256 → 256)
  → 256-dim body_vector
```

**What it captures:** posture (open/closed), fidgeting, self-touching gestures, movement energy.

---

### 3.2 Pipeline 2 — Face

**Pretrained models:** MediaPipe Face Mesh + DeepFace  
**Input:** Video frames  
**Raw output:** 1404 landmark features + 512 DeepFace features

```
Face Mesh (1404) + DeepFace (512) per frame
  → Concatenate → Linear(1916 → 384)
  → LSTM (1 layer, 384 hidden)
  → Last hidden state
  → Linear(384 → 384)
  → 384-dim face_vector
```

**What it captures:** facial expressions, eye contact, blink rate, smile authenticity.

---

### 3.3 Pipeline 3 — Voice

**Pretrained models:** Wav2Vec 2.0 (Meta) + Classical handcrafted features  
**Input:** 5-second audio clip  
**Raw output:** 1536-dim Wav2Vec (mean+std pooled) + 166-dim classical = **1702-dim voice vector**

```
Wav2Vec 2.0 (facebook/wav2vec2-base)
  → last_hidden_state (T' × 768)
  → mean pooling → 768-dim
  → std  pooling → 768-dim
  → Concatenate  → 1536-dim w2v_meanstd

Classical features (librosa)
  → MFCCs (13) × {mean, std, Δ-mean, Δ-std, ΔΔ-mean, ΔΔ-std} = 78
  → Pitch stats (mean, std, min, max, range, median, skew, kurtosis, slope) + voiced_ratio = 10
  → Energy stats (mean, std, min, max, range, skew, kurtosis, slope) = 8
  → Spectral (centroid, bandwidth, rolloff, flatness) × {mean, std} = 8
  → Spectral contrast (7 bands) × {mean, std} = 14
  → ZCR × {mean, std} = 2
  → Chroma (12) × {mean, std} = 24
  → Tonnetz (6) × {mean, std} = 12
  → Tempo + onset_rate = 2
  → Jitter + shimmer + HNR = 3
  → Pause stats (count, rate, mean_dur, max_dur, total_ratio) = 5
  → Total: 166-dim classical_vec

Concatenate [w2v_meanstd (1536) | classical_vec (166)] → 1702-dim
  → Linear(1702 → 512) → ReLU → Dropout(0.2)
  → Linear(512 → 512)
  → 512-dim voice_vector
```

> ⚠️ **Implementation notes:**
> - Wav2Vec uses **mean+std pooling** (not mean-only), producing 1536-dim. The encoder input must be `Linear(1702 → 512)`.
> - `Wav2Vec2FeatureExtractor` applies z-score normalisation internally (`do_normalize=True`). Do **not** peak-normalise the waveform before passing it to the processor.
> - Classical features use their own peak-normalisation and pre-emphasis internally (see `features_classical.py`). Pass the raw waveform; the function handles it.
> - The batched Wav2Vec API (`extract_wav2vec_features_batch`) uses attention-mask-aware pooling so padded frames don't pollute mean/std — use this during training for speed.
> - Max clip length is capped at 20s inside `features_wav2vec.py` to prevent OOM.

**What it captures:** pitch patterns, voice tremor, speech rate, vocal energy, jitter/shimmer/HNR, spectral texture, rhythmic onset patterns.

---

### 3.4 Pipeline 4 — Text

**Pretrained models:** Whisper (transcription) + BERT (understanding)  
**Input:** 5-second audio clip  
**Raw output:** BERT [CLS] embedding = 768-dim

```
BERT [CLS] token (768)
  → Linear(768 → 512) → ReLU → Dropout(0.2)
  → Linear(512 → 512)
  → 512-dim text_vector
```

**Silence / Empty Transcription Handling**

If Whisper produces an empty string (silent window), the standard BERT [CLS] token is meaningless. Two options:

| Option | Approach |
|---|---|
| A — Simple | Replace text vector with a **zero vector** (512-dim); optionally set a binary flag for the fusion layer |
| B — Learned *(default)* | Train a **learnable silence embedding** used whenever transcription is empty; updated normally during backprop |

> **Default: Option B.** After Whisper transcription, check `if text.strip() == ""`; if so, emit the silence embedding directly; otherwise pass through BERT and encoder as usual.

**What it captures:** sentiment, word choice patterns, hesitation language.

---

## 4. Fusion Layer

Simple concatenation with an MLP — no cross-attention complexity.

```
body_vec  (256)  ─┐
face_vec  (384)  ─┤
voice_vec (512)  ─┼─→ Concat (1664)
text_vec  (512)  ─┘
                      → Linear(1664 → 768) → ReLU → Dropout(0.3)
                      → Linear(768 → 512)  → ReLU
                      → 512-dim fused_vector
```

The fused vector is **512-dim** to preserve information from the larger pipelines.

---

## 5. Judgment Heads

Each head is a small 2-layer MLP. Every model has its **own complete copy** of pipelines + fusion + head.

---

### 5.1 Head 1 — Arousal & Valence *(Regression)*

Based on Russell's Circumplex Model of affect.

| Dimension | Range | Meaning |
|---|---|---|
| Arousal | −1 (calm) → +1 (excited) | Energy / activation level |
| Valence | −1 (negative) → +1 (positive) | Pleasantness / mood |

**The 2D Emotion Space**

```
                 HIGH AROUSAL (+1)
                       │
       Angry ●         │         ● Excited
       Tense ●         │         ● Happy
                       │
NEGATIVE (−1) ─────────┼───────────── POSITIVE (+1)
VALENCE                │                   VALENCE
                       │
       Sad   ●         │         ● Calm
       Bored ●         │         ● Relaxed
                       │
                 LOW AROUSAL (−1)
```

**Architecture**

```
fused_vector (512)
  → Linear(512 → 128) → ReLU → Dropout(0.2)
  → Linear(128 → 64)  → ReLU
  → Linear(64 → 2)
  → Tanh               ← squash to [−1, 1]
```

**Loss Function**

```
Primary:     MSE Loss on both outputs
             L = MSE(arousal_pred, arousal_true) + MSE(valence_pred, valence_true)

Recommended: CCC Loss (Concordance Correlation Coefficient)
             CCC = (2 · ρ · σ_pred · σ_true) / (σ_pred² + σ_true² + (μ_pred − μ_true)²)

Combined:    L_total = 0.5 · MSE + 0.5 · CCC
```

**Evaluation metrics:** MSE / RMSE, MAE, Pearson Correlation, CCC  
**Target performance:** CCC > 0.5 (good), CCC > 0.7 (excellent)

**Label Acquisition Strategy**

| Dataset | Native Labels | Strategy |
|---|---|---|
| IEMOCAP | Continuous arousal/valence (1–5 scale) | Normalise: `(value − 3) / 2` → [−1, +1] |
| CMU-MOSEI | Discrete emotion intensities + sentiment | Pseudo-arousal = max(happy, angry, surprised, fearful); valence = sentiment / 3 |
| RAVDESS / CREMA-D | Categorical emotion labels | Convert using mapping table below |

**Emotion → Arousal/Valence Mapping**

| Emotion | Arousal | Valence |
|---|---|---|
| Happy | +0.6 | +0.8 |
| Excited | +0.9 | +0.7 |
| Calm | −0.5 | +0.5 |
| Neutral | 0.0 | 0.0 |
| Sad | −0.5 | −0.7 |
| Angry | +0.8 | −0.7 |
| Fearful | +0.7 | −0.6 |
| Surprised | +0.7 | +0.2 |
| Disgusted | +0.3 | −0.8 |
| Bored | −0.6 | −0.3 |

**Output example**

```json
{
  "arousal": 0.72,
  "valence": -0.45,
  "interpretation": "High energy, negative mood (likely angry/stressed)"
}
```

**Recommended visualizations:**
- Scatter plot of predicted vs. true arousal/valence
- 2D emotion map of prediction distribution
- Trajectory plot over time for a full video
- Comparison with categorical Emotion head

---

### 5.2 Head 2 — Interview Assessment *(Big Five Regression)*

Predicts five continuous personality traits from the First Impressions V2 dataset (ChaLearn).

| Trait | Description |
|---|---|
| Openness to Experience | Creativity, curiosity |
| Conscientiousness | Organization, dependability |
| Extraversion | Sociability, energy |
| Agreeableness | Compassion, cooperativeness |
| Neuroticism | Emotional instability |

**Architecture**

```
fused_vector (512)
  → Linear(512 → 128) → ReLU → Dropout(0.2)
  → Linear(128 → 5)
  → Linear activation  ← unbounded regression outputs
```

> Outputs are **unbounded real numbers** (z-scores). Training uses MSE directly on the original scale to avoid saturation. If bounded outputs are preferred for interpretability, normalise labels to [0, 1] and use Sigmoid — but all labels must be strictly within that range.

**Loss:** MSE across all five outputs  
**Evaluation:** Pearson correlation per trait (standard First Impressions V2 metric)  
**Target:** correlation > 0.4 per trait

**Why regression over classification?**
- ✅ Direct use of First Impressions V2 labels — zero conversion
- ✅ No threshold ambiguity or conflicting rules
- ✅ Regression is the established benchmark
- ✅ Easier to defend in a thesis

**Output example**

```json
{
  "interview_assessment": {
    "openness": 0.62,
    "conscientiousness": -0.10,
    "extraversion": 0.85,
    "agreeableness": 0.41,
    "neuroticism": -0.23
  }
}
```

---

### 5.3 Head 3 — Emotional State *(7 Classes)*

**Classes:** Happy, Sad, Angry, Fearful, Surprised, Disgusted, Neutral

**Architecture**

```
fused_vector (512)
  → Linear(512 → 128) → ReLU → Dropout(0.2)
  → Linear(128 → 7)
  → Softmax
```

**Loss:** CrossEntropy (weighted if class imbalance exists)  
**Datasets:** RAVDESS, IEMOCAP, CMU-MOSEI, CREMA-D

> Note: Image-only datasets (FER2013, AffectNet) are not used — they don't fit the video pipeline.

---

### 5.4 Head 4 — Engagement Level *(4 Classes)*

| Class | Value | Description |
|---|---|---|
| Not Engaged | 0 | No attention |
| Barely Engaged | 1 | Minimal attention |
| Engaged | 2 | Normal attention |
| Highly Engaged | 3 | Active participation |

> ⚠️ DAiSEE provides 4 labels per clip (Boredom, Engagement, Confusion, Frustration, each 0–3). This head uses **only the Engagement column** as the class label. The other three are ignored. Do **not** treat it as multi-label; the encoder expects a **single integer class index**.

**Architecture**

```
fused_vector (512)
  → Linear(512 → 128) → ReLU → Dropout(0.2)
  → Linear(128 → 4)
  → Softmax
```

**Loss:** CrossEntropy (weighted — DAiSEE is imbalanced toward class 2)  
**Datasets:** DAiSEE (primary), EmotiW (secondary)

---

## 6. Training Strategy

### 6.1 Independent Training

Each model is trained completely separately.

| Model | Training Data | Loss |
|---|---|---|
| Arousal/Valence | IEMOCAP + CMU-MOSEI + mapped RAVDESS, CREMA-D | MSE / CCC |
| Interview | First Impressions V2 + MIT Interview | MSE (Big Five regression) |
| Emotion | RAVDESS + CREMA-D + IEMOCAP + CMU-MOSEI | CrossEntropy (weighted) |
| Engagement | DAiSEE + EmotiW | CrossEntropy (4 classes, weighted) |

### 6.2 Transfer Learning

| Component | Training |
|---|---|
| MediaPipe Pose | ❌ Frozen |
| MediaPipe Face Mesh | ❌ Frozen |
| DeepFace | ❌ Frozen |
| Wav2Vec 2.0 | ❌ Frozen |
| Whisper | ❌ Frozen |
| BERT | ❌ Frozen |
| 4 Custom Encoders | ✅ Trained (per model) |
| Fusion Layer | ✅ Trained (per model) |
| Judgment Head | ✅ Trained (per model) |

> **Trainable parameters per model: ~3–4 million**

### 6.3 Training Loop (Per Model)

```
1. Cache pretrained features for all relevant videos (run once, save to disk)

2. Train model:
   For each epoch:
     For each batch:
       - Load cached features
       - Pass through encoders → fusion → head
       - Compute loss
       - Backprop through encoders + fusion + head

3. Validate after each epoch; save best checkpoint
```

### 6.4 Recommended Build Order

```
1. Emotion Model      ← easiest, most studied
2. Engagement Model   ← straightforward
3. Interview Model    ← rich dataset
4. Arousal/Valence    ← regression, slightly trickier evaluation
```

### 6.5 Data Augmentation

| Pipeline | Augmentation |
|---|---|
| Body | Random keypoint dropout (10%) |
| Face | Horizontal flip |
| Voice | Pitch shift ±5%, add noise |
| Text | Random word dropout (10%) |

### 6.6 Hyperparameters

| Parameter | Classification Models | Regression Models (A/V & Interview) |
|---|---|---|
| Optimizer | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 |
| Batch size | 16 | 16 |
| Epochs | 30–50 | 30–50 |
| Dropout | 0.2–0.3 | 0.2–0.3 |
| Loss | CrossEntropy | MSE / CCC |
| Output activation | Softmax | Tanh (A/V) · Linear (Interview) |

### 6.7 Training Enhancements

#### a) CCC Loss for Arousal/Valence

Directly optimises the preferred evaluation metric; more robust to scale/location shifts than MSE alone.

```python
def ccc_loss(y_pred, y_true):
    # y_pred, y_true: (batch, 2)  →  [arousal, valence]
    loss = 0.0
    for i in range(2):
        pred = y_pred[:, i]
        true = y_true[:, i]
        mu_pred  = torch.mean(pred)
        mu_true  = torch.mean(true)
        var_pred = torch.var(pred)
        var_true = torch.var(true)
        cov      = torch.mean((pred - mu_pred) * (true - mu_true))
        denom    = var_pred + var_true + (mu_pred - mu_true) ** 2
        loss    += 1.0 - (2.0 * cov + 1e-8) / (denom + 1e-8)
    return loss
```

Combine with MSE for stability: `L_total = 0.5 · MSE + 0.5 · CCC`

#### b) Subject-Independent Splits

For datasets with speaker IDs (IEMOCAP, RAVDESS, DAiSEE), enforce a **subject-independent train/val/test split** — no individual appears in more than one split. This prevents memorising person-specific cues and gives an honest generalisation estimate.

#### c) Weighted Loss for Class Imbalance

Heads 3 and 4 suffer from class imbalance ("Neutral" dominates in Emotion; "Engaged" dominates in DAiSEE). Use weighted CrossEntropy with weights inversely proportional to class frequency, and oversample minority classes.

```python
weights  = 1.0 / class_frequencies
weights  = weights / weights.sum() * num_classes
loss_fn  = nn.CrossEntropyLoss(weight=weights)
```

#### d) Gradient Clipping

Stabilises training with LSTM encoders.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### e) Mixed Precision *(optional)*

Use PyTorch AMP if the GPU supports it — reduces memory footprint and allows larger batches.

#### f) Early Stopping

Monitor the primary validation metric (CCC for A/V, Pearson for Interview, accuracy for Emotion/Engagement). Stop training if no improvement for **10 epochs**; save the best checkpoint.

#### g) Silent Window Handling

During feature caching, mark windows with empty transcriptions. During training, the text encoder returns the **learned silence embedding** and backprop updates it normally.

---

## 7. Datasets

### Arousal & Valence (Head 1)

| Dataset | Size | Native A/V Labels | Cost |
|---|---|---|---|
| CMU-MOSEI | 23,453 sentences | ⚠️ Derived (pseudo-arousal + normalised sentiment) | Free |
| IEMOCAP | 10,039 samples | ✅ Continuous (1–5 scale) | Free (research) |
| RAVDESS | 7,356 files | ⚠️ Derived from emotion mapping | Free |
| CREMA-D | 7,442 clips | ⚠️ Derived from emotion mapping | Free |

### Interview Assessment (Head 2)

| Dataset | Size | Cost |
|---|---|---|
| First Impressions V2 (ChaLearn) | 10,000 videos | Free |
| MIT Interview Dataset | Varies | Free |

### Emotional State (Head 3)

| Dataset | Size | Cost |
|---|---|---|
| RAVDESS | 7,356 files | Free |
| IEMOCAP | 10,039 samples | Free (research) |
| CMU-MOSEI | 23,453 sentences | Free |
| CREMA-D | 7,442 clips | Free |

### Engagement Level (Head 4)

| Dataset | Size | Cost |
|---|---|---|
| DAiSEE | 9,068 videos | Free |
| EmotiW | Varies | Free |

> **💡 Minimal starting set (recommended):**
> - Arousal/Valence → IEMOCAP (primary) + CMU-MOSEI (supplementary)
> - Interview → First Impressions V2
> - Emotion → RAVDESS + CREMA-D
> - Engagement → DAiSEE

---

## 8. Video Processing

### Sliding Window Parameters

| Parameter | Value |
|---|---|
| Window Size | 5 seconds |
| Step Size | 3 seconds (2-second overlap) |
| Frames per Window | ~150 (at 30 fps) |

### Processing Pipeline

```
1. Load video file
2. Split into 5-second windows (3-second stride)
3. For each window:
   a. Extract frames → MediaPipe Pose + Face Mesh + DeepFace
   b. Extract audio  → Wav2Vec 2.0
   c. Transcribe     → Whisper → BERT (or silence embedding)
   d. Run all 4 models on cached features
4. Collect per-window predictions
5. Aggregate per task:
   - Timeline   : per-window predictions
   - Overall    : majority vote / averaged probabilities (classification)
                  averaged values + trajectory (regression)
```

### Speed Tips

- **Cache pretrained features once per video** — massive speedup, never recompute MediaPipe/Wav2Vec/BERT per epoch
- **Batch process windows on GPU** — 5–10× faster
- **Use Whisper `tiny` or `base`** — 3× faster than `large`
- **Run all 4 models on the same cached features** — zero redundant computation

### Estimated Processing Times

| Video Length | Estimated Time |
|---|---|
| 5 minutes | 1–2 minutes |
| 30 minutes | 5–10 minutes |
| 1 hour | 15–25 minutes |

---

## 9. Output Format

### Per-Window Output (JSON)

```json
{
  "window_id": 12,
  "timestamp": [60.0, 65.0],
  "predictions": {
    "arousal_valence": {
      "arousal": 0.72,
      "valence": -0.45,
      "interpretation": "High arousal, negative valence"
    },
    "interview": {
      "openness": 0.62,
      "conscientiousness": -0.10,
      "extraversion": 0.85,
      "agreeableness": 0.41,
      "neuroticism": -0.23
    },
    "emotion": {
      "label": "Happy",
      "confidence": 0.89
    },
    "engagement": {
      "label": "Engaged",
      "confidence": 0.78
    }
  }
}
```

### Full Video Report

- Timeline graphs for all 4 categories
- Arousal & Valence as two line plots over time
- 2D arousal-valence trajectory plot (emotional journey)
- Overall summary per category
- Average arousal/valence and Big Five profile across the full video
- Most frequent emotion / engagement labels with confidence scores

**Export formats:** JSON · CSV · PDF

---

## 10. Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- FFmpeg

### Setup

```bash
git clone https://github.com/yourusername/trivex.git
cd trivex
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
mediapipe>=0.10.0
deepface>=0.0.79
openai-whisper>=20230918
librosa>=0.10.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 11. Usage

**Step 1 — Cache Features** *(run once per video)*

```bash
python cache_features.py --video path/to/video.mp4 --output cache/
```

**Step 2 — Train Each Model** *(run separately)*

```bash
python train_arousal_valence.py --config configs/arousal_valence.yaml
python train_interview.py       --config configs/interview.yaml
python train_emotion.py         --config configs/emotion.yaml
python train_engagement.py      --config configs/engagement.yaml
```

**Step 3 — Analyze a Video** *(uses all 4 models)*

```bash
python analyze.py --video path/to/video.mp4 --output results/
```

**Python API**

```python
from trivex import TrivexAnalyzer

analyzer = TrivexAnalyzer(
    arousal_valence_model = "checkpoints/arousal_valence.pt",
    interview_model       = "checkpoints/interview.pt",
    emotion_model         = "checkpoints/emotion.pt",
    engagement_model      = "checkpoints/engagement.pt",
)

results = analyzer.analyze("interview.mp4")
print(results.summary)
print(f"Average arousal: {results.arousal_valence.mean_arousal:.2f}")
print(f"Average valence: {results.arousal_valence.mean_valence:.2f}")

results.export_json("results.json")
results.plot_arousal_valence_trajectory("av_plot.png")
```

---

## 12. Project Structure

```
trivex/
├── README.md
├── requirements.txt
├── configs/
│   ├── arousal_valence.yaml
│   ├── interview.yaml
│   ├── emotion.yaml
│   └── engagement.yaml
├── data/
│   ├── datasets/             # Raw datasets
│   └── cache/                # Cached features
├── trivex/
│   ├── __init__.py
│   ├── pipelines/
│   │   ├── body.py
│   │   ├── face.py
│   │   ├── voice.py
│   │   └── text.py
│   ├── encoders/
│   │   ├── body_encoder.py
│   │   ├── face_encoder.py
│   │   ├── voice_encoder.py
│   │   └── text_encoder.py
│   ├── fusion.py
│   ├── heads/
│   │   ├── arousal_valence.py
│   │   ├── interview.py
│   │   ├── emotion.py
│   │   └── engagement.py
│   ├── models/
│   │   ├── arousal_valence_model.py
│   │   ├── interview_model.py
│   │   ├── emotion_model.py
│   │   └── engagement_model.py
│   ├── datasets/
│   │   ├── cmu_mosei.py
│   │   ├── iemocap.py
│   │   ├── ravdess.py
│   │   ├── crema_d.py
│   │   ├── first_impressions.py
│   │   └── daisee.py
│   ├── losses/
│   │   ├── ccc_loss.py
│   │   └── weighted_ce.py
│   ├── utils/
│   │   ├── label_mapping.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── analyze.py
├── scripts/
│   ├── cache_features.py
│   └── prepare_datasets.py
├── train_arousal_valence.py
├── train_interview.py
├── train_emotion.py
├── train_engagement.py
├── analyze.py
└── checkpoints/
```

---

## 13. Technical Requirements

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | GTX 1660 6 GB | RTX 3060 12 GB |
| RAM | 16 GB | 32 GB |
| Storage | 100 GB | 250 GB SSD |

> No GPU? Google Colab's free tier works for training small models.

### Software

| Package | Purpose |
|---|---|
| Python 3.9+ | Programming language |
| PyTorch 2.0+ | Deep learning framework |
| MediaPipe | Body pose, face mesh |
| DeepFace | Face appearance features |
| HuggingFace Transformers | Wav2Vec 2.0, BERT |
| OpenAI Whisper | Speech-to-text |
| OpenCV | Video processing |

---

## 14. Development Roadmap

| Phase | Tasks | Duration |
|---|---|---|
| Phase 1 — Setup | Install dependencies, test pretrained models | 3 days |
| Phase 2 — Data | Download datasets, organise, get research access | 7 days |
| Phase 3 — Feature Caching | Run pretrained models on all videos, save features | 5 days |
| Phase 4 — Shared Code | Build pipelines, encoders, fusion (reusable code) | 5 days |
| Phase 5 — Emotion Model | Build, train, evaluate Emotion model | 6 days |
| Phase 6 — Engagement Model | Build, train, evaluate Engagement model | 5 days |
| Phase 7 — Interview Model | Build, train, evaluate Interview model | 5 days |
| Phase 8 — Arousal/Valence Model | Build, train, evaluate A/V regression model | 6 days |
| Phase 9 — Integration | Combine all 4 models into unified inference | 3 days |
| Phase 10 — Video Pipeline | Sliding window + aggregation + 2D plots | 4 days |
| Phase 11 — Polish | Visualization, evaluation, documentation | 5 days |
| **Total** | | **~54 days** |

---

## Dimension Flow Summary

```
Body:    132 → 256 → 256  ─┐
Face:   1916 → 384 → 384  ─┤
Voice:  1702 → 512 → 512  ─┼─→ Concat (1664) → MLP → 512 → HEAD
Text:    768 → 512 → 512  ─┘
         (w2v mean+std 1536 + classical 166)

                    512-dim fused vector
                          │
          ┌───────────────┼───────────────┐
          │               │               │               │
          ▼               ▼               ▼               ▼
  Arousal/Valence    Interview         Emotion        Engagement
  (2 outputs, Tanh)  (5 outputs,    (7 classes,     (4 classes,
                      Linear)        Softmax)         Softmax)
```

---

## Tips for Success

- **Start small:** get one pipeline + one head working end-to-end before scaling up.
- **Cache features:** never re-run MediaPipe / Wav2Vec / BERT every epoch.
- **Pick 1–2 datasets per model** initially; expand only after baseline is working.
- **Track experiments** with Weights & Biases or a simple CSV log.
- **Visualise early and often** — scatter plots and confusion matrices catch bugs fast.
- **For Arousal/Valence:** CCC scatter plots look impressive in a thesis and communicate model quality clearly.
- **Build incrementally:** each working model is a defensible milestone.

---

## Expected Performance

| Model | Realistic Target | Stretch Goal |
|---|---|---|
| Emotion | 65–75% accuracy | 80% |
| Engagement | 60–70% accuracy | 75% |
| Interview | Big Five corr > 0.4 | corr > 0.5 |
| Arousal/Valence | CCC > 0.5 | CCC > 0.7 |

---

## License

MIT License. Pretrained models retain their original licenses.

---

## Acknowledgments

Google (MediaPipe, BERT) · Meta (Wav2Vec 2.0) · OpenAI (Whisper) · DeepFace community · All dataset providers.

*A graduation project in multimodal deep learning.* 🎓
