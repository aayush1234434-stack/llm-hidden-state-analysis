from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "probe_default.yaml"


@dataclass
class ModelConfig:
    name: str
    revision: str = "main"


@dataclass
class DatasetConfig:
    id: str
    config: str
    split: str
    revision: str = "main"
    num_samples: int = 500


@dataclass
class GenerationConfig:
    max_new_tokens: int = 24
    prompt_template: str = (
        "Answer the following question in a few words.\nQuestion: {question}\nAnswer:"
    )


@dataclass
class ProbeConfig:
    n_splits: int = 10
    test_size: float = 0.2
    val_size: float = 0.2
    n_permutations: int = 50
    bootstrap_samples: int = 2000
    audit_sample_size: int = 25


@dataclass
class OutputConfig:
    results_dir: Path = field(default_factory=lambda: REPO_ROOT / "results")
    assets_dir: Path = field(default_factory=lambda: REPO_ROOT / "assets")


@dataclass
class ExperimentConfig:
    seed: int = 42
    model: ModelConfig = field(default_factory=lambda: ModelConfig("Qwen/Qwen2-1.5B-Instruct"))
    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            id="mandarjoshi/trivia_qa",
            config="rc.nocontext",
            split="validation",
        )
    )
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    config_path: Path | None = None

    def ensure_dirs(self) -> None:
        self.output.results_dir.mkdir(parents=True, exist_ok=True)
        self.output.assets_dir.mkdir(parents=True, exist_ok=True)

    def split_seeds(self) -> list[int]:
        return list(range(self.seed, self.seed + self.probe.n_splits))

    def cv_fold_seeds(self) -> list[int]:
        return self.split_seeds()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_config(path: Path | str | None = None) -> ExperimentConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    model_raw = raw.get("model", {})
    dataset_raw = raw.get("dataset", {})
    generation_raw = raw.get("generation", {})
    probe_raw = raw.get("probe", {})
    output_raw = raw.get("output", {})

    model_revision = os.environ.get("MODEL_REVISION", model_raw.get("revision", "main"))
    dataset_revision = os.environ.get(
        "DATASET_REVISION", dataset_raw.get("revision", "main")
    )

    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        model=ModelConfig(
            name=model_raw.get("name", "Qwen/Qwen2-1.5B-Instruct"),
            revision=model_revision,
        ),
        dataset=DatasetConfig(
            id=dataset_raw.get("id", "mandarjoshi/trivia_qa"),
            config=dataset_raw.get("config", "rc.nocontext"),
            split=dataset_raw.get("split", "validation"),
            revision=dataset_revision,
            num_samples=int(dataset_raw.get("num_samples", 500)),
        ),
        generation=GenerationConfig(
            max_new_tokens=int(generation_raw.get("max_new_tokens", 24)),
            prompt_template=generation_raw.get(
                "prompt_template",
                "Answer the following question in a few words.\nQuestion: {question}\nAnswer:",
            ),
        ),
        probe=ProbeConfig(
            n_splits=int(probe_raw.get("n_splits", 10)),
            test_size=float(probe_raw.get("test_size", 0.2)),
            val_size=float(probe_raw.get("val_size", 0.2)),
            n_permutations=int(probe_raw.get("n_permutations", 50)),
            bootstrap_samples=int(probe_raw.get("bootstrap_samples", 2000)),
            audit_sample_size=int(probe_raw.get("audit_sample_size", 25)),
        ),
        output=OutputConfig(
            results_dir=_resolve_path(output_raw.get("results_dir", "results")),
            assets_dir=_resolve_path(output_raw.get("assets_dir", "assets")),
        ),
        config_path=config_path,
    )


def add_config_args(parser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="YAML experiment config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results output directory",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=None,
        help="Override assets output directory",
    )


def apply_cli_overrides(cfg: ExperimentConfig, args) -> ExperimentConfig:
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "output_dir", None) is not None:
        cfg.output.results_dir = _resolve_path(args.output_dir)
    if getattr(args, "assets_dir", None) is not None:
        cfg.output.assets_dir = _resolve_path(args.assets_dir)
    if getattr(args, "config", None) is not None:
        cfg.config_path = Path(args.config).resolve()
    cfg.ensure_dirs()
    return cfg
