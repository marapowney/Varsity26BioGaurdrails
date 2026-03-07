#!/usr/bin/env python3
"""
Build a BRCA1 probe dataset for safe/unsafe classification.

This script implements the first concrete step of the linear-probe plan:
1) load BRCA1 variant annotations,
2) apply a strict label policy,
3) construct fixed-length reference and alternate windows,
4) create leakage-aware group splits.

Default behavior is intentionally conservative:
- safe: benign/likely benign OR FUNC
- unsafe: pathogenic/likely pathogenic OR LOF
- excluded: uncertain/conflicting/absent if no strong evidence
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PATHOGENIC_TERMS = {
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
}
BENIGN_TERMS = {
    "benign",
    "likely benign",
}


@dataclass(frozen=True)
class LabelDecision:
    label: str | None
    source: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BRCA1 safe/unsafe subsequence dataset for a linear probe."
    )
    parser.add_argument(
        "--variants-xlsx",
        type=Path,
        default=Path("notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx"),
        help="Path to the BRCA1 variant spreadsheet.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=Path,
        default=Path("notebooks/brca1/GRCh37.p13_chr17.fna.gz"),
        help="Path to chromosome reference FASTA (gzipped or plain text).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("probe/data/processed/brca1_v1"),
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8192,
        help="Fixed DNA window size around variant position.",
    )
    parser.add_argument(
        "--group-bin-size",
        type=int,
        default=1000,
        help="Bin size for split grouping (reduces neighborhood leakage).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Validation split fraction at group level.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Test split fraction at group level.",
    )
    parser.add_argument(
        "--include-int-as-safe",
        action="store_true",
        help="If set, classify INT variants as safe instead of excluding.",
    )
    parser.add_argument(
        "--num-random-safe-windows",
        type=int,
        default=0,
        help="Optional number of additional random reference-only safe windows.",
    )
    parser.add_argument(
        "--expected-chromosome",
        type=str,
        default="17",
        help="Expected chromosome in the spreadsheet and FASTA context.",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Downsample the majority class per split so safe and unsafe counts are equal.",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Optional cap for debugging/smoke tests.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def normalize_chromosome(value: Any) -> str:
    chrom = normalize_text(value).lower()
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    return chrom


def read_first_fasta_record(path: Path) -> tuple[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    header = ""
    sequence_chunks: list[str] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    break
                header = line[1:]
                continue
            if header:
                sequence_chunks.append(line.upper())
    if not header or not sequence_chunks:
        raise ValueError(f"No FASTA record found in {path}")
    return header, "".join(sequence_chunks)


def label_variant(row: pd.Series, include_int_as_safe: bool) -> LabelDecision:
    clinvar_simple = normalize_text(row.get("clinvar_simple")).lower()
    func_class = normalize_text(row.get("func.class")).upper()

    if clinvar_simple in PATHOGENIC_TERMS:
        return LabelDecision("unsafe", "clinvar_simple", clinvar_simple)
    if func_class == "LOF":
        return LabelDecision("unsafe", "func.class", "LOF")
    if clinvar_simple in BENIGN_TERMS:
        return LabelDecision("safe", "clinvar_simple", clinvar_simple)
    if func_class == "FUNC":
        return LabelDecision("safe", "func.class", "FUNC")
    if include_int_as_safe and func_class == "INT":
        return LabelDecision("safe", "func.class", "INT_as_safe")
    return LabelDecision(None, "exclude", f"clinvar={clinvar_simple or 'NA'},func={func_class or 'NA'}")


def extract_reference_window(sequence: str, pos_1_based: int, window_size: int) -> str:
    center_index = window_size // 2
    start_1_based = pos_1_based - center_index
    end_1_based = start_1_based + window_size - 1

    left_pad = max(0, 1 - start_1_based)
    right_pad = max(0, end_1_based - len(sequence))

    start_idx = max(1, start_1_based) - 1
    end_idx = min(len(sequence), end_1_based)
    core = sequence[start_idx:end_idx]

    window = ("N" * left_pad) + core + ("N" * right_pad)
    if len(window) != window_size:
        raise RuntimeError("Window length mismatch after padding.")
    return window


def mutate_center_base(ref_window: str, window_size: int, alt_base: str) -> str:
    center_index = window_size // 2
    chars = list(ref_window)
    chars[center_index] = alt_base
    return "".join(chars)


def split_groups(
    group_ids: Iterable[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, str]:
    groups = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    if n_groups < 3:
        return {g: "train" for g in groups}

    n_test = int(round(n_groups * test_fraction))
    n_val = int(round(n_groups * val_fraction))
    n_test = max(1, n_test)
    n_val = max(1, n_val)
    if n_test + n_val >= n_groups:
        n_val = max(1, n_groups - n_test - 1)
    if n_test + n_val >= n_groups:
        n_test = max(1, n_groups - n_val - 1)

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test : n_test + n_val])
    assignment: dict[str, str] = {}
    for group in groups:
        if group in test_groups:
            assignment[group] = "test"
        elif group in val_groups:
            assignment[group] = "val"
        else:
            assignment[group] = "train"
    return assignment


def save_dataframe_prefer_parquet(df: pd.DataFrame, path_stem: Path) -> Path:
    parquet_path = path_stem.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = path_stem.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.window_size < 8:
        raise ValueError("--window-size must be >= 8")
    if not (0.0 <= args.val_fraction < 1.0 and 0.0 <= args.test_fraction < 1.0):
        raise ValueError("--val-fraction and --test-fraction must be in [0,1)")
    if args.val_fraction + args.test_fraction >= 0.95:
        raise ValueError("val+test fractions too large; train split would be too small.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    variants_df = pd.read_excel(args.variants_xlsx, header=2)
    if args.max_variants is not None:
        variants_df = variants_df.head(args.max_variants).copy()

    fasta_header, reference_seq = read_first_fasta_record(args.reference_fasta)
    expected_chrom = normalize_chromosome(args.expected_chromosome)
    if expected_chrom and expected_chrom not in fasta_header.lower():
        print(
            f"WARNING: expected chromosome '{expected_chrom}' not found in FASTA header: {fasta_header}"
        )

    examples: list[dict[str, Any]] = []
    skip_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()

    for _, row in variants_df.iterrows():
        gene = normalize_text(row.get("gene"))
        chrom = normalize_chromosome(row.get("chromosome"))
        if chrom != expected_chrom:
            skip_counts["unexpected_chromosome"] += 1
            continue

        pos_raw = row.get("position (hg19)")
        try:
            pos = int(pos_raw)
        except Exception:
            skip_counts["invalid_position"] += 1
            continue
        if pos < 1 or pos > len(reference_seq):
            skip_counts["position_out_of_bounds"] += 1
            continue

        ref = normalize_text(row.get("reference")).upper()
        alt = normalize_text(row.get("alt")).upper()
        if len(ref) != 1 or len(alt) != 1:
            skip_counts["non_snv_variant"] += 1
            continue
        if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
            skip_counts["invalid_allele"] += 1
            continue

        ref_base = reference_seq[pos - 1]
        if ref_base != ref:
            skip_counts["reference_mismatch"] += 1
            continue

        decision = label_variant(row, include_int_as_safe=args.include_int_as_safe)
        if decision.label is None:
            skip_counts[f"label_excluded:{decision.reason}"] += 1
            continue

        ref_window = extract_reference_window(reference_seq, pos, args.window_size)
        alt_window = mutate_center_base(ref_window, args.window_size, alt)
        pair_id = f"{chrom}:{pos}:{ref}>{alt}"

        base_record = {
            "pair_id": pair_id,
            "gene": gene,
            "chromosome": chrom,
            "position_hg19": pos,
            "reference": ref,
            "alt": alt,
            "func_class": normalize_text(row.get("func.class")),
            "clinvar_simple": normalize_text(row.get("clinvar_simple")),
            "clinvar": normalize_text(row.get("clinvar")),
            "function_score_mean": row.get("function.score.mean"),
            "consequence": normalize_text(row.get("consequence")),
            "label_source": decision.source,
            "label_reason": decision.reason,
        }

        # Alternate allele example (inherits variant label).
        alt_record = dict(base_record)
        alt_record["sample_id"] = f"{pair_id}:alt"
        alt_record["example_kind"] = "alt"
        alt_record["label"] = decision.label
        alt_record["label_int"] = 1 if decision.label == "unsafe" else 0
        alt_record["sequence"] = alt_window
        alt_record["group_id"] = f"{chrom}:{pos // args.group_bin_size}"
        examples.append(alt_record)
        label_counts[decision.label] += 1

        # Reference context example is always safe.
        ref_record = dict(base_record)
        ref_record["sample_id"] = f"{pair_id}:ref"
        ref_record["example_kind"] = "ref"
        ref_record["label"] = "safe"
        ref_record["label_int"] = 0
        ref_record["sequence"] = ref_window
        ref_record["group_id"] = f"{chrom}:{pos // args.group_bin_size}"
        examples.append(ref_record)
        label_counts["safe"] += 1

    # Optional additional random normal windows.
    if args.num_random_safe_windows > 0:
        rng = random.Random(args.seed)
        for i in range(args.num_random_safe_windows):
            pos = rng.randint(1, len(reference_seq))
            ref_window = extract_reference_window(reference_seq, pos, args.window_size)
            pair_id = f"{expected_chrom}:{pos}:normal"
            examples.append(
                {
                    "sample_id": f"{pair_id}:{i}",
                    "pair_id": pair_id,
                    "gene": "NA",
                    "chromosome": expected_chrom,
                    "position_hg19": pos,
                    "reference": reference_seq[pos - 1],
                    "alt": "",
                    "func_class": "",
                    "clinvar_simple": "",
                    "clinvar": "",
                    "function_score_mean": None,
                    "consequence": "normal_reference_window",
                    "label_source": "synthetic_normal",
                    "label_reason": "random_reference_window",
                    "example_kind": "normal",
                    "label": "safe",
                    "label_int": 0,
                    "sequence": ref_window,
                    "group_id": f"{expected_chrom}:normal:{pos // args.group_bin_size}",
                }
            )
            label_counts["safe"] += 1

    if not examples:
        raise RuntimeError("No examples were generated. Check filters and inputs.")

    dataset_df = pd.DataFrame(examples)
    dataset_df = dataset_df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    group_to_split = split_groups(
        dataset_df["group_id"].tolist(),
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    dataset_df["split"] = dataset_df["group_id"].map(group_to_split)
    if dataset_df["split"].isna().any():
        raise RuntimeError("Found examples without split assignment.")

    if args.balance:
        rng = random.Random(args.seed)
        balanced_parts = []
        for split_name, split_df in dataset_df.groupby("split"):
            safe_df = split_df[split_df["label"] == "safe"]
            unsafe_df = split_df[split_df["label"] == "unsafe"]
            n_keep = min(len(safe_df), len(unsafe_df))
            if len(safe_df) > n_keep:
                safe_df = safe_df.sample(n=n_keep, random_state=rng.randint(0, 2**31))
            if len(unsafe_df) > n_keep:
                unsafe_df = unsafe_df.sample(n=n_keep, random_state=rng.randint(0, 2**31))
            balanced_parts.append(pd.concat([safe_df, unsafe_df]))
        dataset_df = pd.concat(balanced_parts).reset_index(drop=True)

    # Save tables.
    full_path = save_dataframe_prefer_parquet(dataset_df, args.output_dir / "dataset_full")
    train_path = save_dataframe_prefer_parquet(
        dataset_df[dataset_df["split"] == "train"].copy(), args.output_dir / "dataset_train"
    )
    val_path = save_dataframe_prefer_parquet(
        dataset_df[dataset_df["split"] == "val"].copy(), args.output_dir / "dataset_val"
    )
    test_path = save_dataframe_prefer_parquet(
        dataset_df[dataset_df["split"] == "test"].copy(), args.output_dir / "dataset_test"
    )

    split_manifest = dataset_df[
        ["sample_id", "pair_id", "group_id", "split", "label", "example_kind"]
    ].copy()
    split_manifest_path = args.output_dir / "splits.csv"
    split_manifest.to_csv(split_manifest_path, index=False)

    summary = {
        "input": {
            "variants_xlsx": str(args.variants_xlsx),
            "reference_fasta": str(args.reference_fasta),
            "window_size": args.window_size,
            "group_bin_size": args.group_bin_size,
            "expected_chromosome": args.expected_chromosome,
            "include_int_as_safe": args.include_int_as_safe,
            "balance": args.balance,
            "num_random_safe_windows": args.num_random_safe_windows,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "max_variants": args.max_variants,
        },
        "counts": {
            "n_examples": int(len(dataset_df)),
            "n_groups": int(dataset_df["group_id"].nunique()),
            "label_counts": dict(Counter(dataset_df["label"])),
            "split_counts": dict(Counter(dataset_df["split"])),
            "split_label_counts": {
                split: dict(Counter(df_split["label"]))
                for split, df_split in dataset_df.groupby("split")
            },
            "skip_counts": dict(skip_counts),
        },
        "outputs": {
            "dataset_full": str(full_path),
            "dataset_train": str(train_path),
            "dataset_val": str(val_path),
            "dataset_test": str(test_path),
            "split_manifest": str(split_manifest_path),
        },
    }

    summary_path = args.output_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = build_dataset(args)
    print("Dataset build complete.")
    print(json.dumps(summary["counts"], indent=2))
    print("Outputs:")
    for key, value in summary["outputs"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
