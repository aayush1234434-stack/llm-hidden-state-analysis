from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
from huggingface_hub import HfApi

from probe.config import ExperimentConfig


def resolve_hub_sha(repo_id: str, revision: str, repo_type: str) -> str:
    try:
        api = HfApi()
        if repo_type == "model":
            info = api.model_info(repo_id, revision=revision)
        else:
            info = api.dataset_info(repo_id, revision=revision)
        return info.sha or revision
    except Exception:
        return revision


def build_run_manifest(
    cfg: ExperimentConfig,
    *,
    command: str,
    device: str,
    dtype,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import datasets
    import sklearn
    import transformers

    manifest = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "config_path": str(cfg.config_path) if cfg.config_path else None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "model_dtype": str(dtype),
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "model": {
            "name": cfg.model.name,
            "revision_requested": cfg.model.revision,
            "revision_resolved": resolve_hub_sha(
                cfg.model.name, cfg.model.revision, "model"
            ),
        },
        "dataset": {
            "id": cfg.dataset.id,
            "config": cfg.dataset.config,
            "split": cfg.dataset.split,
            "revision_requested": cfg.dataset.revision,
            "revision_resolved": resolve_hub_sha(
                cfg.dataset.id, cfg.dataset.revision, "dataset"
            ),
            "num_samples": cfg.dataset.num_samples,
        },
        "seeds": {
            "random_seed": cfg.seed,
            "logistic_regression": cfg.seed,
            "cv_split_seeds": cfg.split_seeds(),
            "bootstrap": cfg.seed,
            "audit_sample": cfg.seed,
        },
        "generation": {
            "max_new_tokens": cfg.generation.max_new_tokens,
            "prompt_template": cfg.generation.prompt_template,
        },
        "probe": {
            "n_splits": cfg.probe.n_splits,
            "test_size": cfg.probe.test_size,
            "val_size": cfg.probe.val_size,
            "n_permutations": cfg.probe.n_permutations,
            "bootstrap_samples": cfg.probe.bootstrap_samples,
            "audit_sample_size": cfg.probe.audit_sample_size,
        },
        "outputs": {
            "results_dir": str(cfg.output.results_dir),
            "assets_dir": str(cfg.output.assets_dir),
        },
    }
    if extra:
        manifest.update(extra)
    return manifest


def save_run_manifest(
    cfg: ExperimentConfig,
    *,
    command: str,
    device: str,
    dtype,
    extra: dict[str, Any] | None = None,
) -> None:
    path = cfg.output.results_dir / "run_manifest.json"
    with open(path, "w") as f:
        json.dump(
            build_run_manifest(cfg, command=command, device=device, dtype=dtype, extra=extra),
            f,
            indent=2,
        )


def load_run_manifest(cfg: ExperimentConfig) -> dict[str, Any]:
    path = cfg.output.results_dir / "run_manifest.json"
    with open(path) as f:
        return json.load(f)
