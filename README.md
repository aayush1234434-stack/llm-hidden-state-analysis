# Can LLMs Predict Their Own Failures?
## Self-Awareness via Internal Circuits

A mechanistic interpretability experiment probing whether correctness information is encoded in the hidden states of a language model before it finishes generating an answer.

---

## Repository Scope

This repository contains two related but separate research tracks:

| Track | Purpose | Entry Point |
|---------|---------|---------|
| Hidden-State Probing (Primary) | Test whether correctness is linearly decodable from layer activations on TriviaQA | `configs/probe_default.yaml` → `python src/demo.py` |
| Gnosis SFT Pipeline (Downstream) | Generate, verify, merge, and fine-tune models that may learn to use internal correctness signals | `DATA_PREPROCESS.md` → `open-r1/` |

The probing experiment is self-contained and only requires:

- `requirements.txt`
- `python src/demo.py`

It does **not** depend on:

- vLLM
- Gnosis training configs
- Data preprocessing scripts

The Gnosis pipeline is downstream infrastructure that investigates a follow-up question:

> If correctness is already encoded in hidden states, can models be trained to use that signal?

### Relationship Between the Two Tracks

#### Probing (This README)

Establishes whether Qwen2 internally encodes answer correctness.

#### Gnosis Data Preparation (`DATA_PREPROCESS.md`)

Builds verified SFT datasets (TriviaQA, Math, etc.) with correctness labels.

#### Gnosis Training (`open-r1/`)

Fine-tunes Qwen3 and related models using SFT/GRPO recipes.

#### Evaluation (`src/evaluation/`)

Measures whether fine-tuned models improve self-aware behavior on held-out benchmarks.

---

## Motivation

When a transformer generates text, it produces hidden-state vectors at every layer.

If a model internally "knows" that it might be wrong, that information may already be encoded in those hidden states before the answer is produced.

This project tests whether answer correctness can be **linearly decoded** from those activations.

Using **Qwen2-1.5B-Instruct**, linear probes are trained on hidden states from every transformer layer to predict whether the model's final answer will be correct.

---

# Experimental Setup

### Dataset

- TriviaQA (`rc.nocontext`, validation split)

### Model

- Qwen2-1.5B-Instruct

### Samples

- 500 questions

### Procedure

1. Run the model on TriviaQA questions.
2. Capture hidden states at the first generated token.
3. Extract vectors from all 28 transformer layers.
4. Score answers using strict TriviaQA Exact Match (EM).
5. Train logistic-regression probes using nested train/validation/test splits (10 folds).
6. Select layers using validation performance only.
7. Compare against:
   - Majority-class baseline
   - Permutation-label baseline

---

# Results

Because the model answers correctly only 12.2% of the time, the dataset is highly imbalanced.

Predicting **"incorrect"** for every sample already achieves **87.8% accuracy**, making raw accuracy a poor metric.

AUROC and balanced accuracy provide a better assessment.

| Metric | Value |
|----------|----------|
| Model accuracy (Strict EM) | 12.2% (61 / 500) |
| Model accuracy (Legacy substring match) | 35.2% (176 / 500) |
| Majority baseline | 87.8% |
| Best validation AUROC | Layer 27 (0.817) |
| Test AUROC | 0.799 [0.770, 0.830] |
| Test balanced accuracy | 0.588 [0.550, 0.625] |
| Test accuracy | 0.876 |
| Permutation AUROC | 0.470 [0.445, 0.495] |
| Permutation balanced accuracy | 0.496 [0.489, 0.503] |

> **Important:** Majority-baseline accuracy (0.878) should not be compared directly against AUROC. AUROC baseline is 0.5 by definition.

---
<img width="1500" height="750" alt="image" src="https://github.com/user-attachments/assets/fdaad8a3-d6b5-4694-b840-a6066da4ea7f" />
 Note: the plot above currently overlays a 0.88 "majority baseline" line on an AUROC axis. Majority-baseline AUROC is 0.5 by definition, not 0.88 — that comparison only holds for plain accuracy. The plot needs a fix (either two y-axes, or drop the majority-baseline line and keep only the permutation-baseline line, which is the correct AUROC comparison).


# Key Observations

## 1. Correctness Is Linearly Decodable

The probe achieves:

- Test AUROC: **0.799**
- Permutation baseline AUROC: **0.470**

This indicates that hidden states contain meaningful information about eventual answer correctness.

Raw accuracy is misleading due to class imbalance.

---

## 2. Signal Concentrates in Late Layers

Validation AUROC rises from approximately:

- Early layers: ~0.64
- Late layers (22–28): 0.767–0.817

The strongest signal appears near output generation.

---

## 3. Early Layers Still Contain Information

Layers 1–13 achieve:

- AUROC ≈ 0.63–0.68

These layers are likely focused on syntax and representation building but still contain non-trivial correctness information.

---

## 4. Largest Transition Occurs Between Layers 21 → 22

Performance increases sharply:

```text
Layer 21: 0.691 AUROC
Layer 22: 0.767 AUROC
```

This is the most abrupt shift observed and warrants further investigation.

---

## 5. Best Layer Is Not Stable

Layer 27 achieved the highest mean validation AUROC.

However:

- It won only 2 of 10 folds.
- Layers 22–28 perform similarly.

The conclusion is:

> Late layers matter, not necessarily Layer 27 specifically.

---

# Limitations

### Severe Class Imbalance

Only 12.2% of answers are correct.

Accuracy is therefore a weak metric.

### Answer Extraction Errors

Manual audits reveal cases where:

- Valid answers exist in generations.
- Extraction logic returns empty or incorrect strings.

This introduces label noise and likely depresses measured performance.

### EM vs Substring-Match Disagreement

The two labeling methods disagree on:

```text
121 / 500 samples (24.2%)
```

This raises questions about label quality.

### Low Alias F1

Mean maximum alias F1:

```text
0.228
```

This suggests answers often fail to closely match gold aliases.

### Unstable Layer Selection

Performance varies across folds.

Results should be interpreted as:

> Late layers contain the strongest signal.

rather than

> Layer 27 is uniquely important.

### Better Models Would Improve Evaluation

A model with approximately 50% accuracy would provide:

- Better class balance
- More interpretable metrics

---

# Future Work

- Fix answer extraction logic
- Re-run labeling pipeline
- Evaluate stronger models
- Test additional datasets
- Probe intermediate reasoning tokens
- Investigate the Layer 21 → 22 transition
- Compare linear vs nonlinear probes
- Correct the AUROC visualization

---

# Repository Structure

```text
src/
├── demo.py
├── collect_activations.py
├── train_probes.py
├── plot_results.py
├── probe/
├── triviaqa_eval.py

configs/
└── probe_default.yaml
```

### Core Files

| File | Purpose |
|--------|--------|
| `demo.py` | Full pipeline orchestrator |
| `collect_activations.py` | Generate answers and save activations |
| `train_probes.py` | Nested CV probe training |
| `plot_results.py` | AUROC visualization |
| `triviaqa_eval.py` | TriviaQA EM/F1 evaluation |

---

# Running the Experiment

See `REPRODUCIBILITY.md` for:

- Exact versions
- Seeds
- Hardware details
- Hub revisions

## Install Dependencies

```bash
pip install -r requirements.txt
pip install torch
```

---

## Run Full Pipeline

```bash
python src/demo.py
```

---

## Run Individual Stages

```bash
python src/collect_activations.py --config configs/probe_default.yaml

python src/train_probes.py --config configs/probe_default.yaml

python src/plot_results.py \
    --config configs/probe_default.yaml \
    --no-show-plot
```

---

## Quick Smoke Test

```bash
python src/demo.py --num-samples 5 --no-show-plot
```

---

## CLI Options

### Available to All Stages

```text
--config
--seed
--output-dir
--assets-dir
```

### Additional Options (`demo.py`)

```text
--model
--num-samples
--stage {all,collect,train,plot}
--no-show-plot
```

---

## Output Locations

```text
results/   # JSON results
assets/    # plots and figures
```

---

## Optional: Pin Hub Revisions

```bash
export MODEL_REVISION=main
export DATASET_REVISION=main

python src/demo.py
```

Each run writes:

```text
results/run_manifest.json
```

containing:

- resolved revisions
- package versions
- seeds
- hardware metadata

---

# Why This Matters

Understanding internal model representations is a central goal of mechanistic interpretability.

If correctness signals exist before answer generation, they may enable:

- Self-verification systems
- Better uncertainty estimation
- More reliable deployment of language models
- Self-aware reasoning architectures
