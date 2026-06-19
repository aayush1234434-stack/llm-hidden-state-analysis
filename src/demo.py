#!/usr/bin/env python3
"""Run the full hidden-state probing pipeline: collect → train → plot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))


def parse_args():
    from probe.config import DEFAULT_CONFIG_PATH, add_config_args

    parser = argparse.ArgumentParser(description=__doc__)
    add_config_args(parser)
    parser.add_argument("--model", default=None, help="Override model from config")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--no-show-plot", action="store_true")
    parser.add_argument(
        "--stage",
        choices=["all", "collect", "train", "plot"],
        default="all",
        help="Run a single stage or the full pipeline",
    )
    parser.set_defaults(config=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def _argv_for(args, script: str) -> list[str]:
    argv = [sys.executable, str(SRC_DIR / script)]
    if args.config:
        argv += ["--config", str(args.config)]
    if args.seed is not None:
        argv += ["--seed", str(args.seed)]
    if args.output_dir:
        argv += ["--output-dir", str(args.output_dir)]
    if args.assets_dir:
        argv += ["--assets-dir", str(args.assets_dir)]
    if script == "collect_activations.py":
        if args.model:
            argv += ["--model", args.model]
        if args.num_samples is not None:
            argv += ["--num-samples", str(args.num_samples)]
    if script == "plot_results.py" and args.no_show_plot:
        argv.append("--no-show-plot")
    return argv


def main():
    args = parse_args()
    stages = {
        "collect": "collect_activations.py",
        "train": "train_probes.py",
        "plot": "plot_results.py",
    }

    if args.stage == "all":
        for script in stages.values():
            print(f"\n=== {script} ===\n")
            subprocess.run(_argv_for(args, script), check=True)
        return

    subprocess.run(_argv_for(args, stages[args.stage]), check=True)


if __name__ == "__main__":
    main()
