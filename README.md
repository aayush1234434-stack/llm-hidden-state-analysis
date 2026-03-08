# Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits

A small mechanistic interpretability experiment probing whether correctness information is encoded in the hidden states of a language model **before it finishes generating an answer**.

## Overview

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
src/            core probing and evaluation code
scripts/        experiment setup scripts
assets/         figures and visualizations
results/        experiment outputs
```

---

## Running the Experiment

Install dependencies:

```
pip install -r requirements.txt
```

Run the probing experiment:

```
python src/demo.py
```

---

## Motivation

Understanding what models internally represent is a core question in **mechanistic interpretability**.

If correctness signals are encoded before an answer is generated, it may enable:

* self-verification systems
* uncertainty estimation
* safer deployment of language models
