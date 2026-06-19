#!/usr/bin/env python3
"""Train layer-wise linear probes with nested CV on collected activations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe.config import add_config_args, apply_cli_overrides, load_config
from probe.manifest import load_run_manifest, save_run_manifest
from probe.probes import (
    bootstrap_ci,
    fit_probe,
    probe_metrics,
    stratified_train_val_test_split,
    summarize_metric,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    return parser.parse_args()


def load_activations(cfg):
    path = cfg.output.results_dir / "activations.npz"
    if not path.exists():
        raise SystemExit(
            f"Missing {path}. Run collect_activations.py first."
        )
    data = np.load(path)
    return (
        data["hidden_states"],
        data["labels"],
        int(data["num_layers"]),
    )


def run(cfg) -> None:
    command = "python src/train_probes.py"
    hidden_states, labels, num_layers = load_activations(cfg)
    y = np.array(labels)

    if len(np.unique(y)) < 2:
        only = int(y[0]) if len(y) else -1
        raise SystemExit(
            f"All {len(y)} samples have label {only}; cannot train probes.\n"
            "Inspect results/results.json and results/label_audit.json."
        )

    manifest = load_run_manifest(cfg)
    device = manifest.get("device", "unknown")
    dtype = manifest.get("model_dtype", "unknown")

    baseline = max(y.mean(), 1 - y.mean())
    split_seeds = cfg.split_seeds()
    n_splits = cfg.probe.n_splits

    layer_val_auroc = np.full((n_splits, num_layers), np.nan)
    layer_test_auroc = np.full((n_splits, num_layers), np.nan)
    layer_test_bal_acc = np.full((n_splits, num_layers), np.nan)
    layer_test_acc = np.full((n_splits, num_layers), np.nan)

    selected_layers = []
    selected_test_auroc = []
    selected_test_bal_acc = []
    selected_test_acc = []

    print(
        f"Training probes ({n_splits} stratified splits, nested train/val/test)..."
    )
    for fold, seed in enumerate(tqdm(split_seeds, desc="CV folds")):
        for layer_idx in range(num_layers):
            X = hidden_states[:, layer_idx, :]
            X_train, X_val, X_test, y_train, y_val, y_test = (
                stratified_train_val_test_split(X, y, seed, cfg)
            )
            probe = fit_probe(X_train, y_train, cfg)
            val_m = probe_metrics(probe, X_val, y_val)
            test_m = probe_metrics(probe, X_test, y_test)
            layer_val_auroc[fold, layer_idx] = val_m["auroc"]
            layer_test_auroc[fold, layer_idx] = test_m["auroc"]
            layer_test_bal_acc[fold, layer_idx] = test_m["balanced_accuracy"]
            layer_test_acc[fold, layer_idx] = test_m["accuracy"]

        best_layer = int(np.nanargmax(layer_val_auroc[fold]))
        selected_layers.append(best_layer)
        selected_test_auroc.append(layer_test_auroc[fold, best_layer])
        selected_test_bal_acc.append(layer_test_bal_acc[fold, best_layer])
        selected_test_acc.append(layer_test_acc[fold, best_layer])

    mean_val_auroc = np.nanmean(layer_val_auroc, axis=0)
    best_layer = int(np.nanargmax(mean_val_auroc))
    layer_freq = np.bincount(selected_layers, minlength=num_layers)

    print("\nLayer-wise mean test AUROC (selected on validation):")
    for i in range(num_layers):
        mean_auroc = np.nanmean(layer_test_auroc[:, i])
        marker = " <-- best (by val AUROC)" if i == best_layer else ""
        print(
            f"  Layer {i+1:>2}: AUROC={mean_auroc:.3f}  "
            f"bal_acc={np.nanmean(layer_test_bal_acc[:, i]):.3f}{marker}"
        )

    print(f"\nBest layer (mean validation AUROC): {best_layer + 1}")
    print(f"Selected on val in {layer_freq[best_layer]}/{n_splits} folds")

    print("\nHeld-out test metrics for validation-selected layer per fold:")
    sel_auroc = summarize_metric("AUROC", selected_test_auroc, cfg)
    sel_bal = summarize_metric("Balanced accuracy", selected_test_bal_acc, cfg)
    sel_acc = summarize_metric("Accuracy", selected_test_acc, cfg)
    print(f"  {'Majority baseline':22s}: {baseline:.3f}")

    perm_test_auroc = []
    perm_test_bal_acc = []
    print(f"\nPermutation baseline ({cfg.probe.n_permutations} shuffles)...")
    perms_per_fold = max(1, cfg.probe.n_permutations // n_splits)
    for fold, seed in enumerate(tqdm(split_seeds, desc="Permutation")):
        X = hidden_states[:, best_layer, :]
        X_train, X_val, X_test, y_train, y_val, y_test = (
            stratified_train_val_test_split(X, y, seed, cfg)
        )
        for perm_idx in range(perms_per_fold):
            perm_seed = seed * 1000 + perm_idx
            probe = fit_probe(
                X_train, y_train, cfg, permute_labels=True, perm_seed=perm_seed
            )
            test_m = probe_metrics(probe, X_test, y_test)
            perm_test_auroc.append(test_m["auroc"])
            perm_test_bal_acc.append(test_m["balanced_accuracy"])

    print(f"Permutation baseline (layer {best_layer + 1}):")
    perm_auroc = summarize_metric("AUROC", perm_test_auroc, cfg)
    perm_bal = summarize_metric("Balanced accuracy", perm_test_bal_acc, cfg)

    validation_summary = {
        "n_splits": n_splits,
        "test_size": cfg.probe.test_size,
        "val_size": cfg.probe.val_size,
        "best_layer": best_layer + 1,
        "best_layer_val_selection_freq": int(layer_freq[best_layer]),
        "majority_baseline": baseline,
        "selected_layer_test": {
            "auroc": sel_auroc,
            "balanced_accuracy": sel_bal,
            "accuracy": sel_acc,
        },
        "permutation_baseline": {
            "auroc": perm_auroc,
            "balanced_accuracy": perm_bal,
            "n_permutations": len(perm_test_auroc),
        },
        "per_layer_mean_test_auroc": [
            float(np.nanmean(layer_test_auroc[:, i])) for i in range(num_layers)
        ],
        "per_layer_mean_test_balanced_accuracy": [
            float(np.nanmean(layer_test_bal_acc[:, i])) for i in range(num_layers)
        ],
    }

    validation_summary["layer_test_auroc_ci"] = {}
    for layer_idx in range(num_layers):
        _, lo, hi = bootstrap_ci(layer_test_auroc[:, layer_idx], cfg)
        validation_summary["layer_test_auroc_ci"][str(layer_idx + 1)] = {
            "low": lo,
            "high": hi,
        }

    out_path = cfg.output.results_dir / "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(validation_summary, f, indent=2)
    print(f"\nSaved {out_path}")

    save_run_manifest(
        cfg,
        command=command,
        device=device,
        dtype=dtype,
        extra={"stage": "train_probes", "validation_summary": validation_summary},
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    run(cfg)


if __name__ == "__main__":
    main()
