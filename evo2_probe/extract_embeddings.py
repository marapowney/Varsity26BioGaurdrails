#!/usr/bin/env python3
"""
Extract frozen Evo2 embeddings for linear probing.

Processes every sequence in the dataset through a frozen Evo2 model and caches
per-sample embeddings (mean-pooled and last-token) to disk. Identical sequences
are deduplicated so the model only runs once per unique window.

Outputs one .npz per split containing:
    sample_ids  : (n_samples,) string array
    labels      : (n_samples,) int array (0=safe, 1=unsafe)
    mean_pool   : (n_samples, hidden_dim) float32
    last_token  : (n_samples, hidden_dim) float32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import sys

import pkgutil
import yaml

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evo2.utils import CONFIG_MAP, MODEL_NAMES


def _num_layers_for_model(model_name: str) -> int:
    """Return the number of transformer blocks for a given Evo2 model."""
    config_path = CONFIG_MAP[model_name]
    config = yaml.safe_load(pkgutil.get_data("evo2", config_path))
    return int(config["num_layers"])


def _valid_layer_names(model_name: str) -> list[str]:
    """Return the list of valid ``blocks.N`` layer names for *model_name*."""
    n = _num_layers_for_model(model_name)
    return [f"blocks.{i}" for i in range(n)]


def _resolve_layer(layer_arg: str, model_name: str) -> str:
    """Normalise and validate the ``--layer`` argument.

    Accepts either ``blocks.N`` or a bare integer ``N``.  Raises
    ``SystemExit`` with a helpful message when the value is out of range.
    """
    if layer_arg.isdigit():
        layer_arg = f"blocks.{layer_arg}"

    valid = _valid_layer_names(model_name)
    if layer_arg not in valid:
        n = _num_layers_for_model(model_name)
        sys.exit(
            f"Error: '{layer_arg}' is not a valid layer for {model_name} "
            f"(num_layers={n}).\n"
            f"Valid layers: blocks.0 .. blocks.{n - 1}"
        )
    return layer_arg


def _parse_layer_spec(spec: str, model_name: str) -> list[str]:
    """Parse a layer specification like ``0-31``, ``0,5,10``, or ``0-10,15``.

    Returns a sorted list of ``blocks.N`` strings.
    """
    valid_names = set(_valid_layer_names(model_name))
    n = _num_layers_for_model(model_name)
    layers: set[str] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        halves = part.split("-", 1)
        if len(halves) == 2 and halves[0].isdigit() and halves[1].isdigit():
            lo, hi = int(halves[0]), int(halves[1])
            for i in range(lo, hi + 1):
                name = f"blocks.{i}"
                if name not in valid_names:
                    sys.exit(
                        f"Error: blocks.{i} out of range for {model_name} "
                        f"(valid: 0..{n - 1})"
                    )
                layers.add(name)
        else:
            layers.add(_resolve_layer(part, model_name))
    return sorted(layers, key=lambda x: int(x.split(".")[1]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract frozen Evo2 embeddings for the probe dataset."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("probe/data/processed/brca1_v1"),
        help="Directory containing dataset_full.parquet from build_dataset.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("probe/features/brca1_v1"),
        help="Base directory for embedding outputs. A subdirectory named "
        "after the layer (e.g. blocks.12/) is created automatically.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="evo2_1b_base",
        choices=MODEL_NAMES,
        help="Evo2 checkpoint name.",
    )
    p.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Layer to extract embeddings from. Accepts 'blocks.N' or just N "
        "(e.g. --layer 12 or --layer blocks.12). Defaults to the last block "
        "for the chosen model. Use --list-layers to see valid values.",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Multiple layers to extract in a single model load, e.g. '0-31' "
        "or '0,5,10,15'. The model is loaded once and all layers are "
        "extracted per sequence in one forward pass. Overrides --layer.",
    )
    p.add_argument(
        "--list-layers",
        action="store_true",
        help="Print valid layer names for the chosen model and exit.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference.",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=200,
        help="Save a checkpoint file every N unique sequences.",
    )
    p.add_argument(
        "--no-flash-attn",
        action="store_true",
        help="Disable flash attention (uses PyTorch SDPA instead). "
        "Auto-detected if flash_attn is not installed.",
    )
    return p.parse_args()


def compute_seq_hash(seq: str) -> str:
    return hashlib.md5(seq.encode()).hexdigest()


def extract_single(
    model,
    tokenizer,
    sequence: str,
    layer_name: str,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one sequence through the model and return (mean_pool, last_token)."""
    input_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
    hidden = embeddings[layer_name]  # (1, seq_len, hidden_dim)

    mean_pool = hidden[0].mean(dim=0).cpu().to(torch.float32).numpy()
    last_token = hidden[0, -1, :].cpu().to(torch.float32).numpy()

    del hidden, embeddings, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mean_pool, last_token


def extract_multi_layer(
    model,
    tokenizer,
    sequence: str,
    layer_names: list[str],
    device: str,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Run one sequence and return ``{layer: (mean_pool, last_token)}`` for all requested layers."""
    input_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).unsqueeze(0).to(device)

    _, embeddings = model(input_ids, return_embeddings=True, layer_names=layer_names)

    results = {}
    for layer_name in layer_names:
        hidden = embeddings[layer_name]  # (1, seq_len, hidden_dim)
        mean_pool = hidden[0].mean(dim=0).cpu().to(torch.float32).numpy()
        last_token = hidden[0, -1, :].cpu().to(torch.float32).numpy()
        results[layer_name] = (mean_pool, last_token)

    del embeddings, input_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def load_checkpoint(checkpoint_path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load partially completed extraction checkpoint."""
    if not checkpoint_path.exists():
        return {}
    data = np.load(checkpoint_path, allow_pickle=True)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    hashes = data["hashes"]
    mean_pool = data["mean_pool"]
    last_token = data["last_token"]
    for i, h in enumerate(hashes):
        cache[str(h)] = (mean_pool[i], last_token[i])
    return cache


def save_checkpoint(
    checkpoint_path: Path,
    cache: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    if not cache:
        return
    hashes = list(cache.keys())
    mean_pool = np.stack([cache[h][0] for h in hashes])
    last_token = np.stack([cache[h][1] for h in hashes])
    np.savez(
        checkpoint_path,
        hashes=np.array(hashes),
        mean_pool=mean_pool,
        last_token=last_token,
    )


def main() -> None:
    args = parse_args()

    # --list-layers: print valid layers and exit
    if args.list_layers:
        valid = _valid_layer_names(args.model_name)
        n = len(valid)
        print(f"Model '{args.model_name}' has {n} layers. Valid --layer values:")
        for name in valid:
            print(f"  {name}")
        sys.exit(0)

    # --- resolve which layers to extract ---
    if args.layers is not None:
        layer_names = _parse_layer_spec(args.layers, args.model_name)
    elif args.layer is not None:
        layer_names = [_resolve_layer(args.layer, args.model_name)]
    else:
        n = _num_layers_for_model(args.model_name)
        layer_names = [f"blocks.{n - 1}"]
        print(f"No layer(s) specified; defaulting to last block: {layer_names[0]}")

    for ln in layer_names:
        (args.output_dir / ln).mkdir(parents=True, exist_ok=True)

    print(f"Layers: {', '.join(layer_names)} ({len(layer_names)} total)")

    # --- load dataset ---
    dataset_path = args.dataset_dir / "dataset_full.parquet"
    if not dataset_path.exists():
        dataset_path = args.dataset_dir / "dataset_full.csv"
    df = (
        pd.read_parquet(dataset_path)
        if dataset_path.suffix == ".parquet"
        else pd.read_csv(dataset_path)
    )
    print(f"Loaded {len(df)} examples from {dataset_path}")

    # --- deduplicate sequences ---
    unique_seqs = df["sequence"].unique()
    seq_hash_map = {s: compute_seq_hash(s) for s in unique_seqs}
    unique_hashes = list({seq_hash_map[s] for s in unique_seqs})
    hash_to_seq = {}
    for s in unique_seqs:
        h = seq_hash_map[s]
        if h not in hash_to_seq:
            hash_to_seq[h] = s
    print(
        f"  {len(df)} total examples → {len(unique_hashes)} unique sequences "
        f"(dedup saves {len(df) - len(unique_hashes)} forward passes)"
    )

    # --- per-layer: check completion & load checkpoints ---
    per_layer_cache: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    layers_to_extract: list[str] = []

    for layer_name in layer_names:
        layer_dir = args.output_dir / layer_name
        if all(
            (layer_dir / f"embeddings_{s}.npz").exists()
            for s in ("train", "val", "test")
        ):
            print(f"  [{layer_name}] Output files exist — skipping.")
            continue

        cache = load_checkpoint(layer_dir / "_checkpoint.npz")
        per_layer_cache[layer_name] = cache
        layers_to_extract.append(layer_name)
        already = sum(1 for h in unique_hashes if h in cache)
        if already:
            print(f"  [{layer_name}] Resuming: {already}/{len(unique_hashes)} cached")

    if not layers_to_extract:
        print("\nAll layers fully extracted — nothing to do.")
        return

    # Sequences that still need at least one layer
    seqs_remaining = [
        h for h in unique_hashes
        if any(h not in per_layer_cache[ln] for ln in layers_to_extract)
    ]

    print(
        f"\n  {len(layers_to_extract)} layer(s) to extract, "
        f"{len(seqs_remaining)} unique sequences need processing"
    )

    # --- load model (once for all layers) ---
    if seqs_remaining:
        print(f"\n{'='*60}")
        print(f"PHASE: Model Loading")
        print(f"{'='*60}")
        disable_flash = args.no_flash_attn
        if not disable_flash:
            try:
                import flash_attn_2_cuda  # noqa: F401
            except ImportError:
                print("  flash_attn not found — falling back to PyTorch SDPA")
                disable_flash = True

        if disable_flash:
            print("  Flash attention DISABLED (using PyTorch SDPA)")

        print(f"Loading model {args.model_name}...", flush=True)
        t0 = time.time()

        if disable_flash:
            import pkgutil
            import yaml
            from evo2.utils import CONFIG_MAP, HF_MODEL_NAME_MAP
            from vortex.model.model import StripedHyena
            from vortex.model.tokenizer import CharLevelTokenizer
            from vortex.model.utils import dotdict, load_checkpoint as vtx_load_ckpt
            from evo2.models import Evo2

            config_path = CONFIG_MAP[args.model_name]
            config = yaml.safe_load(pkgutil.get_data("evo2", config_path))
            config["use_flash_attn"] = False

            try:
                import transformer_engine  # noqa: F401
            except ImportError:
                if config.get("use_fp8_input_projections", False):
                    print("  Transformer Engine not found — disabling FP8 projections (using bf16)")
                    config["use_fp8_input_projections"] = False

            config = dotdict(config)

            evo2_wrapper = Evo2.__new__(Evo2)
            evo2_wrapper.tokenizer = CharLevelTokenizer(512)

            hf_model_name = HF_MODEL_NAME_MAP[args.model_name]
            import huggingface_hub
            from huggingface_hub import snapshot_download, constants, hf_hub_download
            import os

            filename = f"{args.model_name}.pt"
            final_weights_path = os.path.join(
                os.path.dirname(constants.HF_HUB_CACHE), filename
            )
            if os.path.exists(final_weights_path):
                weights_path = final_weights_path
                hf_hub_download(repo_id=hf_model_name, filename="config.json")
            else:
                repo_dir = snapshot_download(repo_id=hf_model_name)
                repo_weights_path = os.path.join(repo_dir, filename)
                if os.path.exists(repo_weights_path):
                    weights_path = repo_weights_path
                else:
                    parts = []
                    part_num = 0
                    while True:
                        part_path = os.path.join(repo_dir, f"{filename}.part{part_num}")
                        if os.path.exists(part_path):
                            parts.append(part_path)
                            part_num += 1
                        else:
                            break
                    if parts:
                        print(f"  Merging {len(parts)} checkpoint shards...")
                        with open(final_weights_path, "wb") as outfile:
                            for part in parts:
                                with open(part, "rb") as infile:
                                    while True:
                                        chunk = infile.read(8192 * 1024)
                                        if not chunk:
                                            break
                                        outfile.write(chunk)
                        weights_path = final_weights_path
                    else:
                        raise FileNotFoundError(
                            f"Could not find {filename} or shards in {repo_dir}"
                        )

            inner_model = StripedHyena(config)
            vtx_load_ckpt(inner_model, weights_path)
            evo2_wrapper.model = inner_model
            model = evo2_wrapper
        else:
            from evo2 import Evo2
            model = Evo2(args.model_name)
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1e9
            mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU memory: {mem_alloc:.1f} / {mem_total:.1f} GB")

        # --- extract embeddings for all layers per sequence ---
        print(f"\n{'='*60}")
        print(f"PHASE: Embedding Extraction")
        print(f"{'='*60}")
        print(f"  Layers: {', '.join(layers_to_extract)}")
        print(f"  Sequences to process: {len(seqs_remaining)}")
        print(f"  Checkpoint interval: every {args.checkpoint_every} sequences")
        sys.stdout.flush()

        hidden_dim_reported = False
        timings: list[float] = []
        extract_start = time.time()

        pbar = tqdm(
            enumerate(seqs_remaining),
            total=len(seqs_remaining),
            desc="Extracting",
            unit="seq",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}, "
                "{postfix}]"
            ),
        )
        for i, h in pbar:
            needed_layers = [
                ln for ln in layers_to_extract
                if h not in per_layer_cache[ln]
            ]
            if not needed_layers:
                continue

            seq = hash_to_seq[h]
            t_seq = time.time()

            if len(needed_layers) == 1:
                mp, lt = extract_single(
                    model, model.tokenizer, seq, needed_layers[0], args.device
                )
                per_layer_cache[needed_layers[0]][h] = (mp, lt)
                if not hidden_dim_reported:
                    hidden_dim_reported = True
                    tqdm.write(f"  Hidden dim: {mp.shape[0]}")
            else:
                results = extract_multi_layer(
                    model, model.tokenizer, seq, needed_layers, args.device
                )
                for ln, (mp, lt) in results.items():
                    per_layer_cache[ln][h] = (mp, lt)
                if not hidden_dim_reported:
                    hidden_dim_reported = True
                    sample_mp = next(iter(results.values()))[0]
                    tqdm.write(f"  Hidden dim: {sample_mp.shape[0]}")

            dt = time.time() - t_seq
            timings.append(dt)

            recent = timings[-50:]
            avg_sec = sum(recent) / len(recent)
            eta_min = avg_sec * (len(seqs_remaining) - i - 1) / 60

            gpu_mb = ""
            if torch.cuda.is_available():
                gpu_mb = f"GPU {torch.cuda.memory_allocated()/1e9:.1f}GB"

            pbar.set_postfix_str(
                f"{avg_sec:.2f}s/seq, ETA {eta_min:.0f}m, {gpu_mb}"
            )

            if (i + 1) % args.checkpoint_every == 0:
                for ln in layers_to_extract:
                    save_checkpoint(
                        args.output_dir / ln / "_checkpoint.npz",
                        per_layer_cache[ln],
                    )
                tqdm.write(
                    f"  [Checkpoint] {i + 1}/{len(seqs_remaining)} sequences "
                    f"({time.time() - extract_start:.0f}s elapsed)"
                )

        pbar.close()
        for ln in layers_to_extract:
            save_checkpoint(
                args.output_dir / ln / "_checkpoint.npz",
                per_layer_cache[ln],
            )

        total_time = time.time() - extract_start
        avg_time = total_time / len(seqs_remaining) if seqs_remaining else 0
        print(f"\n  Extraction complete:")
        print(f"    Total time:  {total_time/60:.1f} min")
        print(f"    Avg speed:   {avg_time:.2f} s/seq")
        print(f"    Throughput:  {len(seqs_remaining)/total_time:.1f} seq/s")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print("All unique sequences already cached — skipping model load.")

    # --- write output files per layer ---
    print(f"\n{'='*60}")
    print(f"PHASE: Writing Output Files")
    print(f"{'='*60}")

    for layer_name in layers_to_extract:
        layer_dir = args.output_dir / layer_name
        cache = per_layer_cache[layer_name]

        sample_hash = next(iter(cache))
        hidden_dim = cache[sample_hash][0].shape[0]
        print(f"\n  [{layer_name}] hidden_dim={hidden_dim}")

        manifest = {
            "model_name": args.model_name,
            "layer": layer_name,
            "hidden_dim": hidden_dim,
            "n_unique_sequences": len(unique_hashes),
            "splits": {},
        }

        for split in ["train", "val", "test"]:
            split_df = df[df["split"] == split].reset_index(drop=True)
            n = len(split_df)

            sample_ids = split_df["sample_id"].values.astype(str)
            labels = split_df["label_int"].values.astype(np.int32)
            mean_pool_arr = np.empty((n, hidden_dim), dtype=np.float32)
            last_token_arr = np.empty((n, hidden_dim), dtype=np.float32)

            for idx, seq in enumerate(split_df["sequence"]):
                h = seq_hash_map[seq]
                mean_pool_arr[idx] = cache[h][0]
                last_token_arr[idx] = cache[h][1]

            out_path = layer_dir / f"embeddings_{split}.npz"
            np.savez(
                out_path,
                sample_ids=sample_ids,
                labels=labels,
                mean_pool=mean_pool_arr,
                last_token=last_token_arr,
            )
            manifest["splits"][split] = {
                "path": str(out_path),
                "n_samples": n,
                "n_safe": int((labels == 0).sum()),
                "n_unsafe": int((labels == 1).sum()),
            }
            print(f"    {split}: {n} samples → {out_path}")

        manifest_path = layer_dir / "feature_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        checkpoint_path = layer_dir / "_checkpoint.npz"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"    Checkpoint removed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
