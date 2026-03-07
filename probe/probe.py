#!/usr/bin/env python3
"""
Run a single input sequence through Evo2 layer 23, extract the embedding,
and predict safe/unsafe using the saved 7B balanced linear probe.
"""

from __future__ import annotations

import argparse
import os
import pickle
import pkgutil
from pathlib import Path

import numpy as np
import torch
import yaml

# Edit this to run with a sequence defined in code (used when no CLI arg given)
INPUT_SEQUENCE = "ACGTACGTACGT"


def extract_embedding(model, tokenizer, sequence: str, layer_name: str, device: str):
    """Extract mean_pool and last_token embeddings from the specified layer."""
    input_ids = (
        torch.tensor(tokenizer.tokenize(sequence), dtype=torch.int)
        .unsqueeze(0)
        .to(device)
    )

    _, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])
    hidden = embeddings[layer_name]  # (1, seq_len, hidden_dim)

    mean_pool = hidden[0].mean(dim=0).cpu().to(torch.float32).numpy()
    last_token = hidden[0, -1, :].cpu().to(torch.float32).numpy()

    return mean_pool, last_token


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract layer-23 embedding and run 7B balanced probe prediction."
    )
    p.add_argument(
        "sequence",
        type=str,
        nargs="?",
        default=None,
        help="Input DNA sequence. If omitted, uses INPUT_SEQUENCE from code.",
    )
    p.add_argument(
        "--probe-path",
        type=Path,
        default=Path("probe/models/brca1_v1_128_7b_balanced_mlp/probe_model.pkl"),
        help="Path to saved probe_model.pkl.",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="evo2_7b",
        help="Evo2 model name for embedding extraction.",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=23,
        help="Layer to extract embeddings from (default: 23).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for model inference.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sequence = args.sequence if args.sequence is not None else INPUT_SEQUENCE
    if not sequence:
        raise SystemExit("No sequence provided. Pass as argument or set INPUT_SEQUENCE in code.")

    layer_name = f"blocks.{args.layer}"
    print(f"Loading Evo2 model: {args.model_name}")

    disable_flash = False
    try:
        import flash_attn_2_cuda  # noqa: F401
    except ImportError:
        print("  flash_attn not found — falling back to PyTorch SDPA")
        disable_flash = True

    if disable_flash:
        from evo2.utils import CONFIG_MAP, HF_MODEL_NAME_MAP
        from evo2.models import Evo2
        from vortex.model.model import StripedHyena
        from vortex.model.tokenizer import CharLevelTokenizer
        from vortex.model.utils import dotdict, load_checkpoint as vtx_load_ckpt
        from huggingface_hub import snapshot_download

        config_path = CONFIG_MAP[args.model_name]
        config = yaml.safe_load(pkgutil.get_data("evo2", config_path))
        config["use_flash_attn"] = False

        try:
            import transformer_engine  # noqa: F401
        except ImportError:
            if config.get("use_fp8_input_projections", False):
                config["use_fp8_input_projections"] = False

        config = dotdict(config)

        evo2_wrapper = Evo2.__new__(Evo2)
        evo2_wrapper.tokenizer = CharLevelTokenizer(512)

        hf_model_name = HF_MODEL_NAME_MAP[args.model_name]
        filename = f"{args.model_name}.pt"
        repo_dir = snapshot_download(repo_id=hf_model_name)
        weights_path = os.path.join(repo_dir, filename)
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Could not find {filename} in {repo_dir}"
            )

        inner_model = StripedHyena(config)
        vtx_load_ckpt(inner_model, weights_path)
        evo2_wrapper.model = inner_model
        model = evo2_wrapper
    else:
        from evo2 import Evo2
        model = Evo2(args.model_name)

    print(f"Extracting embedding from layer {layer_name}...")
    mean_pool, last_token = extract_embedding(
        model, model.tokenizer, sequence, layer_name, args.device
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading probe from {args.probe_path}")
    with open(args.probe_path, "rb") as f:
        probe_data = pickle.load(f)

    scaler = probe_data["scaler"]
    clf = probe_data["model"]
    config = probe_data.get("config", {})

    feature_type = config.get("feature_type", "mean_pool")
    if feature_type == "mean_pool":
        X = mean_pool.reshape(1, -1)
    elif feature_type == "last_token":
        X = last_token.reshape(1, -1)
    elif feature_type == "both":
        X = np.concatenate([mean_pool, last_token]).reshape(1, -1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0, 1]  # P(unsafe)
    pred = clf.predict(X_scaled)[0]

    label = "unsafe" if pred == 1 else "safe"
    print(f"\nPrediction: {label}")
    print(f"P(unsafe): {proba:.4f}")


if __name__ == "__main__":
    main()
