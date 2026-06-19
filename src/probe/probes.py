from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from probe.config import ExperimentConfig


def stratified_train_val_test_split(
    X, y, seed: int, cfg: ExperimentConfig
):
    test_size = cfg.probe.test_size
    val_size = cfg.probe.val_size
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_frac,
        random_state=seed,
        stratify=y_trainval,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def fit_probe(
    X_train,
    y_train,
    cfg: ExperimentConfig,
    *,
    permute_labels: bool = False,
    perm_seed: int | None = None,
):
    y_fit = np.array(y_train, copy=True)
    if permute_labels:
        rng = np.random.default_rng(perm_seed)
        y_fit = rng.permutation(y_fit)
    probe = LogisticRegression(max_iter=1000, random_state=cfg.seed, C=1.0)
    probe.fit(X_train, y_fit)
    return probe


def probe_metrics(probe, X, y) -> dict[str, float]:
    y_pred = probe.predict(X)
    y_prob = probe.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
    }
    try:
        metrics["auroc"] = roc_auc_score(y, y_prob)
    except ValueError:
        metrics["auroc"] = float("nan")
    return metrics


def bootstrap_ci(
    values,
    cfg: ExperimentConfig,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(cfg.seed)
    boot_means = [
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(cfg.probe.bootstrap_samples)
    ]
    alpha = (1.0 - ci) / 2.0
    return (
        float(values.mean()),
        float(np.percentile(boot_means, 100 * alpha)),
        float(np.percentile(boot_means, 100 * (1 - alpha))),
    )


def summarize_metric(name: str, values, cfg: ExperimentConfig) -> dict[str, float]:
    mean, lo, hi = bootstrap_ci(values, cfg)
    print(f"  {name:22s}: {mean:.3f}  [{lo:.3f}, {hi:.3f}]  (n={len(values)})")
    return {"mean": mean, "ci_low": lo, "ci_high": hi}
