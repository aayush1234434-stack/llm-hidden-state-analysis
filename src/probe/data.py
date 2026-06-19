from __future__ import annotations

from datasets import load_dataset

from probe.config import ExperimentConfig


def load_triviaqa_dataset(cfg: ExperimentConfig):
    ds = cfg.dataset
    try:
        dataset = load_dataset(
            ds.id,
            ds.config,
            split=ds.split,
            revision=ds.revision,
        )
    except ValueError as exc:
        if "Invalid pattern" in str(exc) and "**" in str(exc):
            raise RuntimeError(
                "Failed to load TriviaQA due to incompatible datasets/fsspec "
                "versions. Fix: pip install -r requirements.txt"
            ) from exc
        raise
    return dataset.select(range(min(ds.num_samples, len(dataset))))
