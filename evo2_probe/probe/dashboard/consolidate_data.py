#!/usr/bin/env python3
"""Consolidate all per-layer metrics.json files into a single dashboard_data.json."""

import json
import math
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

DIR_TO_KEY = {
    "brca1_v1_128_7b_balanced": "linear_full",
    "brca1_v1_128_7b_balanced_pca95": "linear_pca95",
    "brca1_v1_128_7b_balanced_mlp": "mlp_full",
    "brca1_v1_128_7b_balanced_mlp_pca95": "mlp_pca95",
}

LABELS = {
    "linear_full": "Linear / Full (4096-dim)",
    "linear_pca95": "Linear / PCA95",
    "mlp_full": "MLP / Full (4096-dim)",
    "mlp_pca95": "MLP / PCA95",
}


def sanitize(obj):
    """Replace Infinity/NaN with None so JSON serialization works."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def strip_thresholds(metrics: dict) -> dict:
    """Remove thresholds arrays from ROC curves to reduce file size."""
    for split in ("val", "test"):
        if split in metrics and "roc_curve" in metrics[split]:
            metrics[split]["roc_curve"].pop("thresholds", None)
    return metrics


def main():
    output = {"probes": {}}

    for dir_name, key in sorted(DIR_TO_KEY.items()):
        model_dir = MODELS_DIR / dir_name
        if not model_dir.is_dir():
            print(f"WARNING: {model_dir} not found, skipping")
            continue

        layers = {}
        for layer_idx in range(32):
            layer_dir = model_dir / f"blocks.{layer_idx}"
            metrics_file = layer_dir / "metrics.json"
            if not metrics_file.exists():
                print(f"WARNING: {metrics_file} not found, skipping")
                continue

            with open(metrics_file) as f:
                raw = f.read()
                raw = raw.replace("Infinity", "1e999")
                data = json.loads(raw)

            data = sanitize(data)
            data = strip_thresholds(data)
            layers[str(layer_idx)] = data

        output["probes"][key] = {"label": LABELS[key], "layers": layers}
        print(f"  {key}: {len(layers)} layers loaded")

    out_path = Path(__file__).resolve().parent / "dashboard_data.json"
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
