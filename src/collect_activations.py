#!/usr/bin/env python3
"""Collect model answers and per-layer hidden states on TriviaQA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe.config import add_config_args, apply_cli_overrides, load_config
from probe.data import load_triviaqa_dataset
from probe.inference import ProbeModel
from probe.manifest import save_run_manifest
from triviaqa_eval import evaluate_prediction


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Override number of dataset samples",
    )
    return parser.parse_args()


def run(cfg) -> None:
    command = "python src/collect_activations.py"
    probe_model = ProbeModel(cfg)
    print(f"Using device: {probe_model.device} (dtype={probe_model.dtype})")
    print(f"Results: {cfg.output.results_dir}\n")
    print(f"Model has {probe_model.num_layers} hidden layers.\n")

    save_run_manifest(
        cfg,
        command=command,
        device=probe_model.device,
        dtype=probe_model.dtype,
        extra={"stage": "collect_activations"},
    )

    print("Loading dataset...")
    dataset = load_triviaqa_dataset(cfg)
    print(
        f"Loaded {len(dataset)} samples from "
        f"{cfg.dataset.id} ({cfg.dataset.config}).\n"
    )

    print("Collecting hidden states...")
    all_hidden = [[] for _ in range(probe_model.num_layers)]
    labels: list[int | None] = []
    results = []

    for sample in tqdm(dataset):
        question = sample["question"]
        answer_obj = sample["answer"]
        try:
            raw_answer, layer_vectors = probe_model.get_answer_and_hidden_states(
                question
            )
            scored = evaluate_prediction(raw_answer, answer_obj)
            correct = scored["exact_match"]
            for layer_idx, vec in enumerate(layer_vectors):
                all_hidden[layer_idx].append(vec)
            labels.append(int(correct))
            results.append({
                "question": question,
                "raw_answer": scored["raw_answer"],
                "extracted_answer": scored["extracted_answer"],
                "ground_truths": scored["ground_truths"],
                "best_matching_alias": scored["best_matching_alias"],
                "exact_match": scored["exact_match"],
                "f1": scored["f1"],
                "legacy_substring_match": scored["legacy_substring_match"],
                "correct": correct,
            })
        except Exception as exc:
            for layer_idx in range(probe_model.num_layers):
                all_hidden[layer_idx].append(None)
            labels.append(None)
            print(f"Skipping sample: {exc}")

    valid_mask = [i for i, label in enumerate(labels) if label is not None]
    labels_clean = np.array([labels[i] for i in valid_mask], dtype=np.int32)

    legacy_labels = [int(r["legacy_substring_match"]) for r in results]
    disagreements = sum(
        1 for r in results if r["legacy_substring_match"] != r["exact_match"]
    )
    mean_f1 = float(np.mean([r["f1"] for r in results])) if results else 0.0

    print(f"\nCollected {len(labels_clean)} valid samples.")
    if len(labels_clean):
        print(
            f"Correct (TriviaQA EM): {labels_clean.sum()} / {len(labels_clean)} "
            f"({100 * labels_clean.mean():.1f}%)"
        )
    print(f"Label disagreements (legacy vs EM): {disagreements}")
    print(f"Mean max F1 over aliases: {mean_f1:.3f}\n")

    results_path = cfg.output.results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path}")

    hidden_states = np.stack(
        [
            np.stack([all_hidden[layer][i] for layer in range(probe_model.num_layers)])
            for i in valid_mask
        ],
        axis=0,
    )
    activations_path = cfg.output.results_dir / "activations.npz"
    np.savez_compressed(
        activations_path,
        hidden_states=hidden_states,
        labels=labels_clean,
        valid_indices=np.array(valid_mask, dtype=np.int32),
        num_layers=probe_model.num_layers,
    )
    print(f"Saved {activations_path}")

    audit_rng = np.random.default_rng(cfg.seed)
    audit_n = min(cfg.probe.audit_sample_size, len(results))
    audit_indices = sorted(audit_rng.choice(len(results), size=audit_n, replace=False))
    audit_records = [{"index": int(i), **results[i]} for i in audit_indices]

    print(f"\nManual audit sample ({audit_n} examples):")
    print("-" * 100)
    for rank, row in enumerate(audit_records, start=1):
        print(f"[{rank:02d}] Q: {row['question'][:70]}")
        print(f"     raw:       {row['raw_answer'][:90]}")
        print(f"     extracted: {row['extracted_answer']}")
        print(f"     gold:      {row['best_matching_alias']}")
        print(
            f"     EM={int(row['exact_match'])}  F1={row['f1']:.2f}  "
            f"legacy={int(row['legacy_substring_match'])}"
        )
        print("-" * 100)

    audit_path = cfg.output.results_dir / "label_audit.json"
    with open(audit_path, "w") as f:
        json.dump(
            {
                "labeling": "TriviaQA official normalization with max-over-aliases EM/F1",
                "audit_sample_size": audit_n,
                "label_disagreements_legacy_vs_em": disagreements,
                "samples": audit_records,
            },
            f,
            indent=2,
        )
    print(f"Saved {audit_path}")

    save_run_manifest(
        cfg,
        command=command,
        device=probe_model.device,
        dtype=probe_model.dtype,
        extra={
            "stage": "collect_activations",
            "labeling": {
                "em_correct": int(labels_clean.sum()) if len(labels_clean) else 0,
                "n_samples": int(len(labels_clean)),
                "mean_f1": mean_f1,
                "legacy_vs_em_disagreements": disagreements,
            },
        },
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.num_samples is not None:
        cfg.dataset.num_samples = args.num_samples
    cfg = apply_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
