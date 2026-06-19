#!/usr/bin/env python3
"""Plot layer-wise probe AUROC from validation_results.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe.config import add_config_args, apply_cli_overrides, load_config
from probe.manifest import load_run_manifest, save_run_manifest


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument(
        "--no-show-plot",
        action="store_true",
        help="Save plot without opening an interactive window",
    )
    return parser.parse_args()


def run(cfg, *, show_plot: bool = True) -> Path:
    command = "python src/plot_results.py"
    val_path = cfg.output.results_dir / "validation_results.json"
    if not val_path.exists():
        raise SystemExit(f"Missing {val_path}. Run train_probes.py first.")

    with open(val_path) as f:
        validation = json.load(f)

    manifest = load_run_manifest(cfg)
    model_name = manifest.get("model", {}).get("name", "model")
    n_samples = manifest.get("dataset", {}).get("num_samples", "?")
    n_splits = validation["n_splits"]

    per_layer_auroc = validation["per_layer_mean_test_auroc"]
    num_layers = len(per_layer_auroc)
    baseline = validation["majority_baseline"]
    perm_mean = validation["permutation_baseline"]["auroc"]["mean"]
    best_layer = validation["best_layer"]

    ci_low = []
    ci_high = []
    ci_data = validation.get("layer_test_auroc_ci", {})
    for layer in range(1, num_layers + 1):
        band = ci_data.get(str(layer), {})
        ci_low.append(band.get("low", per_layer_auroc[layer - 1]))
        ci_high.append(band.get("high", per_layer_auroc[layer - 1]))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, num_layers + 1)
    mean_test_auroc = np.array(per_layer_auroc)

    ax.plot(
        x, mean_test_auroc, marker="o", markersize=5,
        linewidth=2, color="#4C8BF5", label="Mean test AUROC",
    )
    ax.fill_between(x, ci_low, ci_high, alpha=0.2, color="#4C8BF5", label="95% bootstrap CI")
    ax.axhline(baseline, linestyle="--", color="#E8453C", linewidth=1.5,
               label=f"Majority baseline ({baseline:.2f})")
    ax.axhline(perm_mean, linestyle="--", color="#9E9E9E", linewidth=1.5,
               label=f"Permutation baseline ({perm_mean:.2f})")
    ax.axvline(best_layer, linestyle=":", color="#34A853", linewidth=1.5,
               label=f"Best layer by val ({best_layer})")
    ax.fill_between(
        x, baseline, mean_test_auroc,
        where=mean_test_auroc > baseline,
        alpha=0.15, color="#4C8BF5", label="Above majority baseline",
    )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Test AUROC", fontsize=12)
    ax.set_title(
        f"Self-Knowledge Probe AUROC by Layer\n"
        f"{model_name} — {n_samples} samples, {n_splits} nested splits",
        fontsize=13,
    )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_xticks(x[::2])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = cfg.output.assets_dir / "layer_probe_accuracy.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved {plot_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    save_run_manifest(
        cfg,
        command=command,
        device=manifest.get("device", "unknown"),
        dtype=manifest.get("model_dtype", "unknown"),
        extra={"stage": "plot_results", "plot_path": str(plot_path)},
    )
    return plot_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)
    run(cfg, show_plot=not args.no_show_plot)


if __name__ == "__main__":
    main()
