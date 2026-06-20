# Reproducibility

This document records pinned dependencies, artifact revisions, seeds, hardware, and exact commands for both tracks in this repository.

---

## 1. Hidden-state probing (`src/demo.py`)

### Environment

| Item | Value |
| ---- | ----- |
| Python | 3.11 |
| Dependencies | [`requirements.txt`](requirements.txt) |
| Install | `pip install -r requirements.txt` then install `torch` for your platform |

**Tested hardware (primary run):** Apple Silicon Mac, `mps`, `torch.bfloat16`  
**Alternative:** `FORCE_DEVICE=cpu python src/demo.py` (slower, fp32)

### Pinned artifacts

| Artifact | ID | Revision | Notes |
| -------- | -- | -------- | ----- |
| Model | `Qwen/Qwen2-1.5B-Instruct` | `main` | Override with `MODEL_REVISION=<sha>` |
| Dataset | `mandarjoshi/trivia_qa` | `main` | Config `rc.nocontext`, split `validation` |
| Dataset subset | First `NUM_SAMPLES=500` rows | — | Fixed order from HF split |

To pin exact Hub commits, set before running:

```bash
export MODEL_REVISION=<hf_commit_sha>
export DATASET_REVISION=<hf_commit_sha>
python src/demo.py
```

Resolved revisions are written to `run_manifest.json` after each run.

### Seeds and splits

| Purpose | Seed / range |
| ------- | ------------ |
| sklearn logistic regression | `42` (`LogisticRegression(random_state=42)`) |
| CV fold splits | `42, 43, …, 51` (`N_SPLITS=10`) |
| Bootstrap CIs | `42` (`BOOTSTRAP_SAMPLES=2000`) |
| Permutation baseline | `fold_seed * 1000 + perm_idx` |
| Manual audit sample | `numpy.random.default_rng(42)` |

### Hyperparameters (see `src/demo.py` CONFIG block)

```
NUM_SAMPLES=500  MAX_NEW_TOKENS=24  N_SPLITS=10
TEST_SIZE=0.2  VAL_SIZE=0.2  N_PERMUTATIONS=50
```

### Exact commands

```bash
pip install -r requirements.txt
pip install torch

python src/demo.py --config configs/probe_default.yaml
python src/demo.py --seed 42 --output-dir results --num-samples 50 --no-show-plot
python src/demo.py --stage train   # resume from saved activations.npz
```

Outputs:

| File | Contents |
| ---- | -------- |
| `results/run_manifest.json` | Versions, hardware, revisions, seeds, config |
| `results/results.json` | Per-sample answers and EM/F1 labels |
| `results/activations.npz` | Hidden states `(n_valid, n_layers, hidden_dim)` + labels |
| `results/label_audit.json` | Manual audit subset |
| `results/validation_results.json` | Nested CV probe metrics + bootstrap CIs |
| `assets/layer_probe_accuracy.png` | Layer-wise test AUROC plot |
| `configs/probe_default.yaml` | Default experiment configuration |

---

## 2. Gnosis data + evaluation pipeline

### Environment

| Item | Value |
| ---- | ----- |
| Python | 3.11 |
| GPU | NVIDIA CUDA (vLLM generation) |
| Data/eval deps | [`gnosis/requirements-gnosis.txt`](gnosis/requirements-gnosis.txt) |
| Full training stack | [`gnosis/scripts/setup_gnosis_env.sh`](gnosis/scripts/setup_gnosis_env.sh) |

The setup script installs:

- `vllm==0.8.5.post1` (brings `torch==2.6.0`)
- Local editable `transformers/` and `trl[vllm]` forks (expected beside repo root)
- `open-r1[dev]` with pins from [`gnosis/open-r1/setup.py`](gnosis/open-r1/setup.py)

```bash
chmod +x gnosis/scripts/setup_gnosis_env.sh
bash gnosis/scripts/setup_gnosis_env.sh
conda activate Gnosis1
export TOKENIZERS_PARALLELISM=false
```

### Pinned training dependencies (`gnosis/open-r1/setup.py`)

| Package | Pin |
| ------- | --- |
| `vllm` | `0.8.5.post1` |
| `torch` | `2.6.0` |
| `transformers` | `4.52.3` (local fork in full setup) |
| `trl[vllm]` | `0.18.0` |
| `math-verify` | `0.5.2` |
| `latex2sympy2_extended` | `>=1.0.6` |
| `pandas` | `>=2.2.3` |
| `accelerate` | `1.4.0` |
| `deepspeed` | `0.16.8` |

### Example commands (TriviaQA SFT shard)

**Generate completions (vLLM):**

```bash
python gnosis/src/data_preprocess/data_generation.py \
  --data_mode hf \
  --dataset_id mandarjoshi/trivia_qa \
  --dataset_config rc \
  --dataset_split train \
  --system_prompt "This is a trivia question. Put your final answer within \\boxed{}." \
  --model_id Qwen/Qwen3-8B \
  --save_dir data/train/Qwen3_8B_trivia_qa40k-6k
```

**Label completions:**

```bash
python gnosis/src/data_preprocess/Label_for_SFT.py \
  --save_dir data/train/Qwen3_8B_trivia_qa40k-6k \
  --task trivia
```

**Merge + rebalance:**

```bash
python gnosis/src/data_preprocess/merge_sft_data.py
```

**SFT training (open-r1):**

```bash
cd gnosis/open-r1
accelerate launch --config_file recipes/accelerate_configs/ddp.yaml \
  src/open_r1/sft.py \
  --config recipes/training/Qwen3/Qwen3-8B_hybrid_gnosis.yaml
```

**Benchmark scoring:**

```bash
python gnosis/src/evaluation/score_completions_Gnosis_outputscores_script_version.py \
  --save_dir data/test/Qwen3_8B_TriviaQA_val \
  --task trivia
```

See [`gnosis/DATA_PREPROCESS.md`](gnosis/DATA_PREPROCESS.md) for full pipeline documentation.

### Gnosis model / dataset revisions

Pin at generation time via CLI:

```bash
python gnosis/src/data_preprocess/data_generation.py \
  --model_id Qwen/Qwen3-8B \
  ...  # add --model_revision <sha> when supported
```

Training YAMLs under `gnosis/open-r1/recipes/training/` reference local dataset paths; record the Hub revision used during `data_generation.py` in your run notes or W&B config.

---

## 3. Recording environment per run

### Probing

`src/demo.py` writes `results/run_manifest.json` automatically, including:

- Python, platform, torch/transformers/datasets versions
- Device and dtype
- Model and dataset Hub revisions (resolved at runtime)
- All config constants and seed values

### Gnosis

For training runs, log:

- `git rev-parse HEAD` (this repo)
- `pip freeze > environment-freeze.txt`
- CUDA device name (`nvidia-smi`)
- W&B run ID (if used)

---

## 4. Known platform notes

| Platform | Probing | Gnosis vLLM |
| -------- | ------- | ----------- |
| Apple Silicon (MPS) | Use `torch.bfloat16`; fp16 produces garbage (`!!!!`) | Not supported |
| Linux + CUDA | `torch.float16` | Primary target |
| CPU | `FORCE_DEVICE=cpu`; very slow | Not practical for vLLM |
