from __future__ import annotations

import os

import torch


def resolve_device() -> str:
    forced = os.environ.get("FORCE_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda", "mps"}:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str):
    # Qwen2 fp16 on MPS produces garbage ("!!!!"); bf16 is fast and stable.
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.bfloat16
    return torch.float32
