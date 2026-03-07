#!/usr/bin/env python3
"""
Create a balanced version of extracted embeddings by downsampling the
majority class (safe) to match the minority class (unsafe) per split.

Reads existing .npz files from extract_embeddings.py and writes new .npz
files with equal safe/unsafe counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Downsample majority class to create balanced embedding splits."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("probe/features/brca1_v1_128"),
        help="Directory with embeddings_{train,val,test}.npz.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("probe/features/brca1_v1_128_balanced"),
        help="Directory for balanced output.",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to balance (default: all).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return p.parse_args()


def balance_split(input_path: Path, output_path: Path, rng: np.random.Generator) -> dict:
    data = np.load(input_path, allow_pickle=True)
    labels = data["labels"]
    sample_ids = data["sample_ids"]
    mean_pool = data["mean_pool"]
    last_token = data["last_token"]

    safe_idx = np.where(labels == 0)[0]
    unsafe_idx = np.where(labels == 1)[0]
    n_safe, n_unsafe = len(safe_idx), len(unsafe_idx)

    n_keep = min(n_safe, n_unsafe)

    if n_safe > n_keep:
        safe_idx = rng.choice(safe_idx, size=n_keep, replace=False)
        safe_idx.sort()
    if n_unsafe > n_keep:
        unsafe_idx = rng.choice(unsafe_idx, size=n_keep, replace=False)
        unsafe_idx.sort()

    keep = np.concatenate([safe_idx, unsafe_idx])
    keep.sort()

    np.savez(
        output_path,
        sample_ids=sample_ids[keep],
        labels=labels[keep],
        mean_pool=mean_pool[keep],
        last_token=last_token[keep],
    )

    return {
        "original_safe": int(n_safe),
        "original_unsafe": int(n_unsafe),
        "kept_safe": int(n_keep if n_safe >= n_keep else n_safe),
        "kept_unsafe": int(n_keep if n_unsafe >= n_keep else n_unsafe),
        "total_kept": int(len(keep)),
        "total_removed": int(len(labels) - len(keep)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest_path = args.input_dir / "feature_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            src_manifest = json.load(f)
    else:
        src_manifest = {}

    summary: dict = {"source": str(args.input_dir), "seed": args.seed, "splits": {}}

    for split in args.splits:
        input_path = args.input_dir / f"embeddings_{split}.npz"
        if not input_path.exists():
            print(f"  Skipping {split}: {input_path} not found")
            continue

        output_path = args.output_dir / f"embeddings_{split}.npz"
        stats = balance_split(input_path, output_path, rng)
        summary["splits"][split] = stats

        print(
            f"  {split:>5s}: {stats['original_safe']}+{stats['original_unsafe']} "
            f"-> {stats['kept_safe']}+{stats['kept_unsafe']} "
            f"(removed {stats['total_removed']} safe examples)"
        )

    new_manifest = {
        **{k: v for k, v in src_manifest.items() if k != "splits"},
        "balanced": True,
        "balance_seed": args.seed,
        "n_unique_sequences": sum(s["total_kept"] for s in summary["splits"].values()),
        "splits": {
            split: {
                "path": str(args.output_dir / f"embeddings_{split}.npz"),
                "n_samples": stats["total_kept"],
                "n_safe": stats["kept_safe"],
                "n_unsafe": stats["kept_unsafe"],
            }
            for split, stats in summary["splits"].items()
        },
    }
    manifest_out = args.output_dir / "feature_manifest.json"
    manifest_out.write_text(json.dumps(new_manifest, indent=2))

    summary_path = args.output_dir / "balance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nManifest: {manifest_out}")
    print(f"Summary:  {summary_path}")


if __name__ == "__main__":
    main()
