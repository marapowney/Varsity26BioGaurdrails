#!/usr/bin/env python3
"""
Reduce embedding dimensionality via PCA, keeping components that explain
a target cumulative variance.

Fits PCA on the training split and applies the same transform to val/test.
Writes new .npz files (same schema as the originals) with reduced features.

Typical usage:
    python pca_embeddings.py \
        --input-dir probe/features/brca1_v1_128_20b_balanced \
        --output-dir probe/features/brca1_v1_128_20b_balanced_pca \
        --variance-threshold 0.95 \
        --feature-type mean_pool \
        --layer 17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PCA-reduce frozen Evo2 embeddings to a target variance threshold."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("probe/features/brca1_v1_128_balanced"),
        help="Directory with embeddings_{train,val,test}.npz.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("probe/features/brca1_v1_128_balanced_pca"),
        help="Directory for PCA-reduced output.",
    )
    p.add_argument(
        "--feature-type",
        choices=["mean_pool", "last_token", "both"],
        default="mean_pool",
        help="Which embedding column(s) to reduce.",
    )
    p.add_argument(
        "--variance-threshold",
        type=float,
        default=0.95,
        help="Keep fewest components that explain at least this fraction of variance (0-1).",
    )
    p.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Hard upper limit on number of components (overrides variance threshold if fewer).",
    )
    p.add_argument(
        "--top-n-print",
        type=int,
        default=30,
        help="Print cumulative variance for the first N components.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    p.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Transformer block (e.g. 17 or blocks.17). "
        "When set, 'blocks.{layer}' is appended to --input-dir and "
        "--output-dir to match the layout from extract_embeddings.py.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def _resolve_layer_dir(base: Path, layer: str | None) -> Path:
    """Append ``blocks.{N}`` to *base* when a layer is specified."""
    if layer is None:
        return base
    layer = layer.strip()
    if layer.isdigit():
        layer = f"blocks.{layer}"
    if not layer.startswith("blocks."):
        raise ValueError(
            f"Invalid --layer value '{layer}'. "
            "Expected an integer or 'blocks.N' (e.g. 17 or blocks.17)."
        )
    return base / layer


def load_features(path: Path, feature_type: str) -> tuple[np.ndarray, dict]:
    """Load an .npz file and return (X, metadata_arrays)."""
    data = np.load(path, allow_pickle=True)

    if feature_type == "mean_pool":
        X = data["mean_pool"]
    elif feature_type == "last_token":
        X = data["last_token"]
    elif feature_type == "both":
        X = np.concatenate([data["mean_pool"], data["last_token"]], axis=1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    meta = {k: data[k] for k in data.files if k not in ("mean_pool", "last_token")}
    return X, meta


def print_variance_table(pca: PCA, top_n: int) -> None:
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    n_show = min(top_n, len(ratios))

    print(f"\n{'PC':>5s}  {'Individual':>11s}  {'Cumulative':>11s}")
    print(f"{'─'*5}  {'─'*11}  {'─'*11}")
    for i in range(n_show):
        print(f"{i+1:5d}  {ratios[i]:11.4%}  {cumulative[i]:11.4%}")
    if n_show < len(ratios):
        print(f"  ... ({len(ratios)} total components)")


def main() -> None:
    args = parse_args()

    args.input_dir = _resolve_layer_dir(args.input_dir, args.layer)
    args.output_dir = _resolve_layer_dir(args.output_dir, args.layer)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── load training split ──────────────────────────────────────────────
    train_path = args.input_dir / "embeddings_train.npz"
    if not train_path.exists():
        raise FileNotFoundError(f"Training file required for PCA fit: {train_path}")

    X_train, meta_train = load_features(train_path, args.feature_type)
    print(f"Loaded training features: {X_train.shape}  (from {train_path})")

    # ── standardise (zero-mean, unit-var) before PCA ─────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    # ── fit full PCA to inspect spectrum ─────────────────────────────────
    n_full = min(X_train_s.shape)
    pca_full = PCA(n_components=n_full, random_state=args.seed)
    pca_full.fit(X_train_s)
    print_variance_table(pca_full, args.top_n_print)

    # ── choose n_components by variance threshold ────────────────────────
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_by_var = int(np.searchsorted(cumulative, args.variance_threshold) + 1)
    n_components = n_by_var
    if args.max_components is not None:
        n_components = min(n_components, args.max_components)
    n_components = min(n_components, n_full)

    actual_var = cumulative[n_components - 1]
    print(
        f"\nVariance threshold: {args.variance_threshold:.2%}"
        f"  ->  keeping {n_components} / {X_train_s.shape[1]} components"
        f"  ({actual_var:.4%} variance explained)"
    )

    # ── refit a compact PCA with the chosen n ────────────────────────────
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(X_train_s)

    # ── transform & save every split ─────────────────────────────────────
    split_info: dict[str, dict] = {}
    for split in args.splits:
        src = args.input_dir / f"embeddings_{split}.npz"
        if not src.exists():
            print(f"  Skipping {split}: {src} not found")
            continue

        X, meta = load_features(src, args.feature_type)
        X_s = scaler.transform(X)
        X_pca = pca.transform(X_s).astype(np.float32)

        dst = args.output_dir / f"embeddings_{split}.npz"
        np.savez(dst, mean_pool=X_pca, **meta)

        split_info[split] = {
            "path": str(dst),
            "original_dim": int(X.shape[1]),
            "pca_dim": int(X_pca.shape[1]),
            "n_samples": int(X_pca.shape[0]),
        }
        print(f"  {split:>5s}: {X.shape} -> {X_pca.shape}  saved {dst}")

    # ── persist PCA artefacts ────────────────────────────────────────────
    pca_artefact = args.output_dir / "pca_model.npz"
    np.savez(
        pca_artefact,
        components=pca.components_,
        mean=pca.mean_,
        explained_variance=pca.explained_variance_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    print(f"\nPCA artefacts saved: {pca_artefact}")

    # ── write manifest ───────────────────────────────────────────────────
    src_manifest_path = args.input_dir / "feature_manifest.json"
    src_manifest = {}
    if src_manifest_path.exists():
        with open(src_manifest_path) as f:
            src_manifest = json.load(f)

    manifest = {
        **{k: v for k, v in src_manifest.items() if k != "splits"},
        "pca": True,
        "variance_threshold": args.variance_threshold,
        "n_components": n_components,
        "variance_explained": float(actual_var),
        "feature_type_reduced": args.feature_type,
        "original_dim": int(X_train.shape[1]),
        "seed": args.seed,
        "splits": split_info,
    }
    manifest_path = args.output_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
