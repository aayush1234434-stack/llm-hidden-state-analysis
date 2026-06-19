# Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits

A mechanistic interpretability experiment probing whether correctness information is encoded in the hidden states of a language model **before it finishes generating an answer**.

## Repository scope

This repo contains **two related but separate tracks**:

| Track | Purpose | Entry point |
| ----- | ------- | ----------- |
| **Hidden-state probing** (primary) | Test whether correctness is linearly decodable from layer activations on TriviaQA | `configs/probe_default.yaml` · `python src/demo.py` |
| **Gnosis SFT pipeline** (downstream) | Generate, verify, merge, and fine-tune models that may learn to use internal correctness signals | [`DATA_PREPROCESS.md`](DATA_PREPROCESS.md) · [`open-r1/`](open-r1/) |

The probing experiment is **self-contained**: it needs only `requirements.txt` and `python src/demo.py`. It does not depend on vLLM, Gnosis training configs, or the data-preprocessing scripts.

The Gnosis pipeline is **downstream research infrastructure** for a follow-on question: *if correctness is already present in hidden states, can we train models to act on it?* Concretely:

1. **Probing** (this README) establishes whether Qwen2 encodes answer correctness in its activations.
2. **Gnosis data prep** ([`DATA_PREPROCESS.md`](DATA_PREPROCESS.md)) builds verified SFT datasets (TriviaQA, math, etc.) with correctness labels from task-specific evaluators.
3. **Gnosis training** ([`open-r1/`](open-r1/)) fine-tunes Qwen3 (and related) models on those datasets via SFT/GRPO recipes under `open-r1/recipes/training/`.
4. **Scoring** ([`src/evaluation/`](src/evaluation/)) evaluates whether fine-tuned models improve at self-aware behavior on held-out benchmarks.

The two tracks share evaluation utilities (e.g. TriviaQA EM/F1 in [`src/triviaqa_eval.py`](src/triviaqa_eval.py) and [`src/evaluation/evaluator.py`](src/evaluation/evaluator.py)) but serve different goals: **measure internal signals** vs **train models to use them**.

---

When a transformer generates text, it produces a hidden state vector at every layer.
If a model internally "knows" it might be wrong, that uncertainty may already be encoded in those hidden states.

This project tests whether **correctness can be linearly decoded from those activations**.

Using **Qwen2-1.5B-Instruct**, I train linear probes on hidden states from every transformer layer to predict whether the model's answer will ultimately be correct.

---

## Experimental Setup

Dataset: TriviaQA (rc.nocontext, validation)
Model: Qwen2-1.5B-Instruct
Samples: 500 questions

Procedure:

1. Run the model on TriviaQA questions.
2. Capture hidden states at the **first generated token**.
3. Extract hidden vectors for all **28 transformer layers**.
4. Train **logistic regression probes** to predict whether the final answer was correct.
5. Evaluate probe accuracy vs layer depth.

If probe accuracy exceeds the **majority baseline**, it suggests correctness information is already encoded in the activations.

---

## Results

| Metric                    | Value                |
| ------------------------- | -------------------- |
| Model accuracy            | 40.2%                |
| Majority baseline         | 60.0%                |
| Best probe accuracy       | **73.0% (Layer 27)** |
| Improvement over baseline | **+13.2%**           |

![Layer Probe Accuracy](assets/layer_probe_accuracy.png)

---

## Key Observations

**1. Correctness is linearly decodable**

A simple logistic regression probe achieves **73% accuracy**, outperforming the majority baseline by **13.2%**.

This suggests correctness information is genuinely encoded in the hidden states.

---

**2. The signal concentrates in late layers**

The strongest signal appears in **layers 27–28**, indicating that the model consolidates answer confidence late in processing.

---

**3. Early and middle layers behave differently**

Layers **3–6** remain near baseline accuracy, suggesting they primarily handle syntactic or positional processing rather than semantic evaluation.

---

**4. A surprising spike appears around layer 15**

Layer 15 shows a noticeable accuracy bump that may indicate an intermediate reasoning stage.

This deserves further investigation.

---

## Limitations

The model answered only **40.2%** of questions correctly, which means the majority class becomes **incorrect answers (60%)**.

This makes the probe's task somewhat easier because predicting "wrong" already performs well.

A stronger experiment would evaluate a model with accuracy closer to **50%**, ensuring balanced classes.

Despite this limitation, the **13% improvement over baseline** suggests a real signal beyond simple class frequency.

---

## Future Work

Possible extensions:

* Repeat the experiment on stronger models
* Evaluate across multiple datasets
* Probe intermediate reasoning tokens rather than only the first generated token
* Investigate the layer-15 spike in more detail
* Compare linear probes with nonlinear probes

---

## Repository Structure

```
src/
  demo.py                  # full pipeline orchestrator
  collect_activations.py   # stage 1: generate answers + save activations
  train_probes.py          # stage 2: nested CV probe training
  plot_results.py          # stage 3: layer AUROC plot
  probe/                   # shared config, inference, probe utilities
  triviaqa_eval.py         # TriviaQA EM/F1 labeling
configs/
  probe_default.yaml       # experiment config (model, dataset, seeds, paths)
```

For the **probing experiment only**, you need `configs/probe_default.yaml`, the `src/probe/` package, and `requirements.txt`. See [Running the experiment](#running-the-experiment) below.

For the **Gnosis SFT pipeline**, see [`DATA_PREPROCESS.md`](DATA_PREPROCESS.md), [`requirements-gnosis.txt`](requirements-gnosis.txt), and [`open-r1/README.md`](open-r1/README.md). That track requires a CUDA GPU, vLLM, and additional setup (`scripts/setup_gnosis_env.sh`).

---

## Running the Experiment

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for pinned versions, Hub revisions, seeds, hardware notes, and exact commands.

Install dependencies:

```
pip install -r requirements.txt
pip install torch   # platform-specific wheel from pytorch.org
```

Run the probing experiment:

```
# Full pipeline (config: configs/probe_default.yaml)
python src/demo.py

# Or stage by stage:
python src/collect_activations.py --config configs/probe_default.yaml
python src/train_probes.py --config configs/probe_default.yaml
python src/plot_results.py --config configs/probe_default.yaml --no-show-plot

# Quick smoke test
python src/demo.py --num-samples 5 --no-show-plot
```

CLI options (all stages): `--config`, `--seed`, `--output-dir`, `--assets-dir`.  
`demo.py` also accepts `--model`, `--num-samples`, `--stage {all,collect,train,plot}`, `--no-show-plot`.

Outputs go to `results/` (JSON) and `assets/` (plots) by default.

Optional: pin Hub artifact revisions before running:

```
export MODEL_REVISION=main
export DATASET_REVISION=main
python src/demo.py
```

Each run writes `results/run_manifest.json` with resolved revisions, package versions, seeds, and hardware.

---

## Motivation

Understanding what models internally represent is a core question in **mechanistic interpretability**.

If correctness signals are encoded before an answer is generated, it may enable:

* self-verification systems
* uncertainty estimation
* safer deployment of language models
