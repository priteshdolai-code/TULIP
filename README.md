# TULIP — Visual Speech Recognition

A deep learning system that reads speech from lip movement alone, with no audio input. Video of a speaker's mouth goes in; the predicted sentence comes out.

This is the implementation behind **"Human–Machine Interaction Through Visual Speech Recognition,"** presented at **ICDICI 2025** — the 6th International Conference on Data Intelligence and Cognitive Informatics.

**Authors:** Anant Chawda, Manjiri Mhatre, Pritesh Dolai, Dr. Ashwini Save — VIVA Institute of Technology, Mumbai.

## The Problem

Visual Speech Recognition is hard because visemes — the visual equivalents of phonemes — are ambiguous. Several different sounds produce near-identical mouth shapes, so a lot of the signal is in subtle, short-lived motion rather than in any single frame. The specific failure mode this work targets is **word substitution**: the model confidently outputs the wrong word because two words looked alike on the lips.

## Approach

An end-to-end sequence-to-sequence pipeline trained with CTC loss:

```
Video input (75 grayscale frames, 46×140, mouth ROI)
   ↓
3D convolutional stack — spatio-temporal feature extraction
   Conv3D(128) → Swish → MaxPool3D(1,2,2)
   Conv3D(256) → Swish → MaxPool3D(1,2,2)
   Conv3D(75)  → Swish → MaxPool3D(1,2,2)
   ↓
Reshape
   ↓
Hybrid temporal backend — BiGRU + BiLSTM (128 features each)
   ↓
Fully connected dense layer over the character set
   ↓
CTC decoding → predicted sentence
```

### Design decisions

**3D convolutions over 2D.** Lip reading depends on how the mouth *moves*, not how it looks in one frame. A 2D CNN applied per frame discards that. Conv3D convolves across height, width and time together, so filters learn motion patterns directly.

**Swish instead of ReLU — the core contribution.** ReLU outputs zero for all negative inputs, which causes dead neurons and discards information in deeper layers. Swish, `s(x) = x · sigmoid(x)`, is smooth and differentiable through the origin and into the negative range. The hypothesis was that this preserves the subtle viseme variations that ReLU's hard cutoff destroys — precisely the small distinctions that cause word substitutions.

It worked. Word substitutions dropped from **0.17 to 0.01** against the ReLU-activated 3DCNN+LSTM model.

**A hybrid BiGRU + BiLSTM backend.** GRU has fewer gates and trains faster; LSTM has more capacity for long-range dependencies. Running both in the temporal backend gave a reduction in training time while only marginally affecting performance relative to a pure LSTM stack.

**Bidirectional throughout.** A viseme is often ambiguous until you see what follows it. Reading the sequence in both directions lets later frames disambiguate earlier ones.

**CTC loss.** Video frames and output characters aren't aligned one-to-one, and hand-producing a frame-level alignment is impractical. CTC lets the network learn the alignment itself from unsegmented sequence pairs.

## Dataset and Training

A curated subset of the **GRID corpus** — 1,000 videos of a subject speaking limited-corpora random 6-word sentences.

| Setting | Value |
|---|---|
| Input | 75 grayscale frames, 46×140 mouth ROI |
| Normalization | Mean-subtracted, divided by standard deviation per frame |
| Split | 80 / 20 train / test |
| Epochs | 100 |
| Dropout | 0.6 |
| Loss | CTC |
| Output | 41-character encoded string |

The dataset is not included in this repository; it is fetched at runtime via `gdown`.

## Results

Evaluated by **Word Error Rate** — substitutions, insertions and deletions against the reference transcript.

| Model | Dataset | WER |
|---|---|---|
| 3DCNN + ReLU + BiGRU (LipNet) | GRID | 4.80% |
| STCNN + BiGRU + Attention | GRID | 3.30% |
| 3DCNN + ReLU + BiLSTM | GRID | 3.26% |
| 3DCNN + Cascaded Attention CTC | GRID | 2.90% |
| **3DCNN + Swish + BiGRU + BiLSTM (this work)** | **GRID** | **2.24%** |

Improvement over prior architectures: **2.56%** over 3DCNN+ReLU+BiGRU, **1.06%** over STCNN+BiGRU+Attention, **1.02%** over 3DCNN+ReLU+BiLSTM, and **0.66%** over 3DCNN+Cascaded Attention CTC.

Word substitutions fell from 0.17 to 0.01 versus the ReLU-activated baseline.

## Tech Stack

Python · TensorFlow / Keras · OpenCV · NumPy · imageio · scikit-image · Matplotlib · Jupyter

## Repository Contents

```
TULIP 3.0.ipynb    main notebook — preprocessing, model, training, evaluation
Demo.ipynb         inference demo
models/            model definitions
frontend/          demo interface
data/              data loading helpers
```

> **Note on this implementation.** The notebook published here implements the
> **3DCNN + ReLU + BiLSTM baseline** (3.26% WER in the table above) — the
> comparison point the proposed model was measured against. The proposed
> **3DCNN + Swish + BiGRU + BiLSTM** architecture and its 2.24% WER result are
> described in the paper.

## Running It

```bash
git clone https://github.com/priteshdolai-code/TULIP.git
cd TULIP
pip install tensorflow opencv-python numpy imageio scikit-image matplotlib gdown
jupyter notebook "TULIP 3.0.ipynb"
```

A GPU is strongly recommended — training on CPU is impractical.

## Recognition

- Presented at **ICDICI 2025** (6th International Conference on Data Intelligence and Cognitive Informatics)
- **1st Rank**, VNPS 2025
- **Best Paper Presentation**, NCRENB 2025

## Limitations and Future Work

- GRID uses a constrained grammar and small vocabulary; performance would drop substantially on unconstrained natural speech. Evaluating on **LRW** or **LRS** is the natural next step.
- Requires a tightly cropped frontal mouth ROI — not robust to pose variation or poor lighting.
- Fixed 75-frame input length.
- Research code in notebook form, not packaged as a deployable service.
- The paper identifies Vision Transformer architectures combined with alternate activation configurations as promising future work.
