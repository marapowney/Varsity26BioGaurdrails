#!/usr/bin/env python3
"""
Train a probe on frozen Evo2 embeddings for safe/unsafe classification.

Loads cached embeddings from extract_embeddings.py and trains either a
logistic regression (linear) or MLP (non-linear) classifier with class
weighting and calibration.  Reports AUROC, AUPRC, F1, precision, recall
on val and test.

Outputs:
    probe_model.pkl     : trained sklearn pipeline
    metrics.json        : full metric suite
    eval_report.txt     : human-readable summary
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a linear probe on frozen Evo2 embeddings."
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("probe/features/brca1_v1_128"),
        help="Directory with embeddings_{train,val,test}.npz from extract_embeddings.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("probe/models/brca1_v1_128"),
        help="Directory for model and evaluation outputs.",
    )
    p.add_argument(
        "--feature-type",
        type=str,
        choices=["mean_pool", "last_token", "both"],
        default="mean_pool",
        help="Which embedding to use as features.",
    )
    p.add_argument(
        "--model",
        type=str,
        choices=["linear", "mlp"],
        default="linear",
        help="Model type: 'linear' (logistic regression) or 'mlp' (multi-layer perceptron).",
    )
    p.add_argument(
        "--C-values",
        type=float,
        nargs="+",
        # default=[0.001,0.01,0.1, 1.0, 10.0, 100.0],
        default=[1.0],
        help="Regularization strengths for linear model.",
    )
    p.add_argument(
        "--mlp-hidden",
        type=str,
        nargs="+",
        # default=["256", "256,128", "512", "512,256"],
        # default=["512,256"],
        default=["32,8"],
        help="Hidden layer sizes to search (comma-separated per config). "
             "Use 'auto' to derive sizes from input dimensionality.",
    )
    p.add_argument(
        "--mlp-alpha",
        type=float,
        nargs="+",
        # default=[1e-4, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1],
        # default=[1e-4],
        default=[1e-2],
        help="L2 regularization strengths for MLP.",
    )
    p.add_argument(
        "--layer",
        type=str,
        required=True,
        help="Transformer block to probe (e.g. 17 or blocks.17). "
        "'blocks.{layer}' is appended to --features-dir and "
        "--output-dir so paths match the layout created by extract_embeddings.py.",
    )
    p.add_argument(
        "--calibrate",
        action="store_true",
        default=True,
        help="Apply Platt scaling calibration on val set.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def load_split(features_dir: Path, split: str, feature_type: str):
    data = np.load(features_dir / f"embeddings_{split}.npz", allow_pickle=True)
    labels = data["labels"]
    sample_ids = data["sample_ids"]

    if feature_type == "mean_pool":
        X = data["mean_pool"]
    elif feature_type == "last_token":
        X = data["last_token"]
    elif feature_type == "both":
        X = np.concatenate([data["mean_pool"], data["last_token"]], axis=1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    return X, labels, sample_ids


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
    }

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    return metrics


def format_report(split_name: str, metrics: dict) -> str:
    cm = metrics["confusion_matrix"]
    lines = [
        f"{'='*50}",
        f"  {split_name.upper()} SET RESULTS",
        f"{'='*50}",
        f"  Samples:   {metrics['n_samples']} ({metrics['n_positive']} unsafe, {metrics['n_negative']} safe)",
        f"",
        f"  AUROC:     {metrics['auroc']:.4f}",
        f"  AUPRC:     {metrics['auprc']:.4f}",
        f"  F1:        {metrics['f1']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall:    {metrics['recall']:.4f}",
        f"  Accuracy:  {metrics['accuracy']:.4f}",
        f"",
        f"  Confusion Matrix:",
        f"                 Predicted",
        f"                 safe  unsafe",
        f"  Actual safe    {cm['tn']:5d}  {cm['fp']:5d}",
        f"  Actual unsafe  {cm['fn']:5d}  {cm['tp']:5d}",
    ]
    return "\n".join(lines)


def _auto_mlp_hidden(input_dim: int) -> list[tuple[int, ...]]:
    """Generate MLP hidden-layer configs scaled to the input dimensionality."""
    def _round(n: int, base: int = 8) -> int:
        return max(base, base * round(n / base))

    d = input_dim
    configs: list[tuple[int, ...]] = [
        (_round(d // 2),),
        (_round(d // 2), _round(d // 4)),
        (_round(d // 4),),
        (_round(d // 4), _round(d // 8)),
    ]
    # deduplicate while preserving order
    seen: set[tuple[int, ...]] = set()
    unique = []
    for c in configs:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _build_candidates(args: argparse.Namespace, input_dim: int) -> list[dict]:
    """Return a list of hyperparameter dicts to search over."""
    if args.model == "linear":
        return [{"C": c} for c in args.C_values]

    if args.mlp_hidden == ["auto"]:
        hidden_configs = _auto_mlp_hidden(input_dim)
    else:
        hidden_configs = [
            tuple(int(x) for x in h.split(",")) for h in args.mlp_hidden
        ]
    return [
        {"hidden": h, "alpha": a}
        for h in hidden_configs
        for a in args.mlp_alpha
    ]


def _make_classifier(model_type: str, params: dict, seed: int):
    """Instantiate a classifier from model type and hyperparams."""
    if model_type == "linear":
        return LogisticRegression(
            C=params["C"],
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed,
        )
    return MLPClassifier(
        hidden_layer_sizes=params["hidden"],
        alpha=params["alpha"],
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=seed,
    )


def _resolve_layer_dir(base: Path, layer: str) -> Path:
    """Append ``blocks.{N}`` to *base*."""
    layer = layer.strip()
    if layer.isdigit():
        layer = f"blocks.{layer}"
    if not layer.startswith("blocks."):
        raise ValueError(
            f"Invalid --layer value '{layer}'. "
            "Expected an integer or 'blocks.N' (e.g. 17 or blocks.17)."
        )
    return base / layer


def main() -> None:
    args = parse_args()

    args.features_dir = _resolve_layer_dir(args.features_dir, args.layer)
    args.output_dir = _resolve_layer_dir(args.output_dir, args.layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.features_dir}")
    print(f"Layer: {args.layer}")
    print(f"Feature type: {args.feature_type}")

    X_train, y_train, ids_train = load_split(args.features_dir, "train", args.feature_type)
    X_val, y_val, ids_val = load_split(args.features_dir, "val", args.feature_type)
    X_test, y_test, ids_test = load_split(args.features_dir, "test", args.feature_type)

    print(f"  Train: {X_train.shape} ({y_train.sum()} unsafe / {len(y_train)} total)")
    print(f"  Val:   {X_val.shape} ({y_val.sum()} unsafe / {len(y_val)} total)")
    print(f"  Test:  {X_test.shape} ({y_test.sum()} unsafe / {len(y_test)} total)")

    # --- hyperparameter search on val set ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    input_dim = X_train_s.shape[1]
    print(f"\nModel: {args.model}  (input dim: {input_dim})")
    candidates = _build_candidates(args, input_dim)
    print(f"Searching over {len(candidates)} configurations...")

    best_auroc = -1.0
    best_params: dict = {}
    best_model = None
    search_results = []

    for params in tqdm(candidates, desc="Hyperparam search"):
        clf = _make_classifier(args.model, params, args.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_s, y_train)

        y_prob_val = clf.predict_proba(X_val_s)[:, 1]
        auroc = roc_auc_score(y_val, y_prob_val)
        auprc = average_precision_score(y_val, y_prob_val)

        result_entry = {**params, "val_auroc": auroc, "val_auprc": auprc}
        search_results.append(result_entry)
        marker = " <-- best" if auroc > best_auroc else ""
        label = "  ".join(f"{k}={v}" for k, v in params.items())
        tqdm.write(f"  {label:<30s}  val AUROC={auroc:.4f}  AUPRC={auprc:.4f}{marker}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_params = params
            best_model = clf

    best_label = "  ".join(f"{k}={v}" for k, v in best_params.items())
    print(f"\nBest: {best_label} (val AUROC={best_auroc:.4f})")

    # --- optional calibration ---
    if args.calibrate:
        print("Calibrating with Platt scaling on val set...")
        calibrated = CalibratedClassifierCV(
            FrozenEstimator(best_model), method="sigmoid"
        )
        calibrated.fit(X_val_s, y_val)
        final_model = calibrated
    else:
        final_model = best_model

    # --- evaluate on val and test ---
    results = {"config": {
        "model": args.model,
        "feature_type": args.feature_type,
        "layer": args.layer,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "calibrated": args.calibrate,
        "seed": args.seed,
        "features_dir": str(args.features_dir),
    }, "search_results": search_results}

    report_lines = []
    roc_data: dict[str, dict] = {}

    eval_splits = [
        ("val", X_val_s, y_val, ids_val),
        ("test", X_test_s, y_test, ids_test),
    ]
    for split_name, X_s, y, ids in tqdm(eval_splits, desc="Evaluating splits", unit="split"):
        y_prob = final_model.predict_proba(X_s)[:, 1]
        y_pred = final_model.predict(X_s)
        metrics = compute_metrics(y, y_prob, y_pred)

        fpr, tpr, thresholds = roc_curve(y, y_prob)
        roc_data[split_name] = {
            "fpr": fpr, "tpr": tpr, "auroc": metrics["auroc"],
        }
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
        }

        results[split_name] = metrics

        report = format_report(split_name, metrics)
        report_lines.append(report)
        print(f"\n{report}")

        errors_path = args.output_dir / f"errors_{split_name}.csv"
        _save_error_cases(errors_path, ids, y, y_prob, y_pred)

    _plot_roc_curves(roc_data, args.output_dir / "roc_curve.png")

    # --- save model ---
    model_path = args.output_dir / "probe_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"scaler": scaler, "model": final_model, "config": results["config"]}, f)
    print(f"\nModel saved: {model_path}")

    # --- save metrics ---
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"Metrics saved: {metrics_path}")

    # --- save report ---
    report_path = args.output_dir / "eval_report.txt"
    report_path.write_text("\n\n".join(report_lines))
    print(f"Report saved: {report_path}")

    print("\nDone.")


def _save_error_cases(
    path: Path,
    sample_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    error_mask = fp_mask | fn_mask

    if not error_mask.any():
        return

    lines = ["sample_id,true_label,predicted_label,unsafe_probability,error_type"]
    for i in np.where(error_mask)[0]:
        etype = "false_positive" if fp_mask[i] else "false_negative"
        lines.append(f"{sample_ids[i]},{y_true[i]},{y_pred[i]},{y_prob[i]:.6f},{etype}")

    path.write_text("\n".join(lines))


def _plot_roc_curves(roc_data: dict[str, dict], output_path: Path) -> None:
    """Plot ROC curves for all evaluated splits and save as PNG."""
    fig, ax = plt.subplots(figsize=(7, 7))

    if "test" in roc_data:
        data = roc_data["test"]
        ax.plot(data["fpr"], data["tpr"], color="#E91E63", lw=2,
                label=f"Test (AUROC = {data['auroc']:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Safe / Unsafe Probe", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve saved: {output_path}")


if __name__ == "__main__":
    main()
