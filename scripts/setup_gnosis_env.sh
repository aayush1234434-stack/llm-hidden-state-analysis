#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------
# Gnosis one-command installer
# Usage (from repo root):
#   chmod +x scripts/setup_gnosis_env.sh
#   bash scripts/setup_gnosis_env.sh
#   conda activate Gnosis
#
# Optional overrides:
#   ENV_NAME=Gnosis PYTHON_VERSION=3.11 VLLM_VERSION=0.8.5.post1 bash scripts/setup_gnosis_env.sh
# ---------------------------------------

ENV_NAME="${ENV_NAME:-Gnosis1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"

# Resolve repo root (assumes this file is in scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Basic sanity checks
for d in transformers trl open-r1 src; do
  [[ -d "$ROOT_DIR/$d" ]] || { echo "❌ Missing '$d/' under repo root: $ROOT_DIR"; exit 1; }
done

# Make conda activation available inside non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
  echo "❌ conda not found. Install Miniconda/Anaconda first, then re-run."
  exit 1
fi
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create env if needed
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "✅ Conda env '${ENV_NAME}' already exists."
else
  echo "🧩 Creating conda env '${ENV_NAME}' (python=${PYTHON_VERSION}) ..."
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

conda activate "${ENV_NAME}"

echo "⬆️  Upgrading pip tooling ..."
python -m pip install --upgrade pip wheel setuptools

# Faster HF uploads (optional, harmless if not installed)
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

echo "📦 Installing Gnosis data/eval requirements ..."
python -m pip install -r "${ROOT_DIR}/requirements-gnosis.txt"

echo "📦 Installing vLLM (${VLLM_VERSION}) ..."
python -m pip install "vllm==${VLLM_VERSION}"

echo "🔎 Checking Torch ..."
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

# flash-attn is optional but recommended for speed
# It may require a proper CUDA toolchain; if it fails, we continue.
echo "⚡ Installing flash-attn (optional) ..."
python -m pip install flash-attn --no-build-isolation || {
  echo "⚠️  flash-attn install failed. Continuing without it."
  echo "    (Common causes: missing CUDA toolchain / incompatible torch+cuda wheels.)"
}

echo "🔧 Installing local Transformers fork (Gnosis-integrated) ..."
python -m pip uninstall -y transformers >/dev/null 2>&1 || true
python -m pip install -e "./transformers"

echo "🔧 Installing local TRL fork ..."
python -m pip install -e "./trl[vllm]"

echo "🔧 Installing open-r1 (dev, no deps) ..."
pushd "open-r1" >/dev/null
GIT_LFS_SKIP_SMUDGE=1 python -m pip install -e ".[dev]" --no-deps
popd >/dev/null

echo "✅ Verifying local installs ..."
python - <<'PY'
import pathlib, transformers, trl
print("transformers →", pathlib.Path(transformers.__file__).resolve())
print("trl          →", pathlib.Path(trl.__file__).resolve())
PY

# Recommended runtime env var
export TOKENIZERS_PARALLELISM=false

cat <<EOF

✅ Setup complete.

Next:
  conda activate ${ENV_NAME}

Tip:
  export TOKENIZERS_PARALLELISM=false

EOF
