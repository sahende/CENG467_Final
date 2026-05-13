# Knowledge Distillation for Task-Specific NLU

**CENG 467 Natural Language Understanding and Generation - Term Project**

İzmir Institute of Technology, Spring 2026

## Project Overview

This project investigates **knowledge distillation (KD)** for compressing BERT-base (109M parameters) into a compact 6-layer student model (67M parameters) on two GLUE benchmark tasks: RTE (Recognizing Textual Entailment) and MRPC (Microsoft Research Paraphrase Corpus).

## Repository Structure

```
├── src/
│   ├── config.py              # Centralized hyperparameters
│   ├── prepare_data.py        # Dataset download and preprocessing
│   ├── models.py              # Model definitions and KD loss function
│   ├── train_baseline.py      # Teacher and baseline student training
│   ├── train_distill.py       # Distilled student training
│   └── evaluate.py            # Model evaluation and comparison
├── results/                   # JSON results for 8 experiments
├── models/                    # Saved model checkpoints
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended, 8GB VRAM minimum)
- Git

### Installation

```bash
git clone https://github.com/sahende/CENG467_Final.git
cd CENG467_Final
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Experiments

### Step 1: Prepare Data

```bash
python src/prepare_data.py
```

Downloads RTE and MRPC from GLUE benchmark, tokenizes with `bert-base-uncased`, and splits into train/validation/test sets.

### Step 2: Train Baseline Models

```bash
python src/train_baseline.py
```

Trains the BERT-base teacher (upper bound) and the 6-layer student without distillation (lower bound) across 8 configurations.

### Step 3: Train Distilled Student

```bash
python src/train_distill.py
```

Trains the 6-layer student using knowledge distillation from the frozen teacher across 8 configurations.

### Step 4: Evaluate and Compare

```bash
python src/evaluate.py
```

Generates comparison tables with Accuracy, Precision, Recall, F1, inference time, and model size.

## Experimental Design

$2^3$ full factorial design (8 configurations):

| Factor | Level 0 | Level 1 |
|:---|:---|:---|
| Weight Decay | 0.00 | 0.01 |
| Sequence Length | 128 | 256 |
| Early Stopping | Disabled | Enabled (patience=3) |

## Key Results (Best Configuration: wd=0.01, seq=128, no ES)

| Model | RTE F1 | MRPC F1 | Parameters | Inference |
|:---|:---:|:---:|:---:|:---:|
| Teacher (BERT-base) | 0.6686 | 0.8279 | 109M | 72.2 ms |
| Student (no KD) | 0.5089 | 0.5927 | 67M | 34.7 ms |
| **Student (Distilled)** | **0.5341** | **0.6242** | **67M** | **35.6 ms** |

**Main Finding:** KD improves performance only with $L_2$ regularization ($\lambda=0.01$). Without weight decay, KD degrades RTE F1 by 4.52 points on average.

## Dependencies

All package versions are pinned in `requirements.txt`. Key dependencies:

- PyTorch >= 2.0.0
- Transformers >= 4.36.0
- Datasets >= 2.15.0
- Scikit-learn >= 1.3.0

## Reproducibility

- All random seeds fixed to 42
- Hyperparameters centralized in `config.py`
- Deterministic train/validation split
- Best model checkpoints saved per epoch
- No API-based models or external services used

## Author

**Şahende Şimşek**

İzmir Institute of Technology, Department of Computer Engineering

Spring 2026