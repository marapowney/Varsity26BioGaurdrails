"""
Phase 0b — Extended dataset from GenBank files

Parses all .gb files in JailbreakDNABench (skipping original/ dirs already
covered by CSV). Extracts CDS features (up to CDS_CAP per genome) and adds
them to the existing 84-sequence CSV dataset.

Output:
  data/processed/sequences_full.npy   (N,) all sequences
  data/processed/labels_full.npy      (N,) labels
  data/processed/families_full.npy    (N,) family names
  data/processed/sources_full.npy     (N,) "csv" or "genbank"

The POC files (sequences.npy etc.) are NOT modified.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from Bio import SeqIO

ROOT      = Path(__file__).resolve().parents[1]
BENCH     = ROOT / "JailbreakDNABench"
PROCESSED = ROOT / "data" / "processed"

TARGET_LEN = 640
MIN_LEN    = 200
CDS_CAP    = 8        # max CDS features to extract per .gb file
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ── helpers (same as Phase 0) ─────────────────────────────────────────────────

def normalise(seq: str, target: int = TARGET_LEN) -> str:
    seq = seq.upper().replace("\n", "").replace(" ", "")
    seq = "".join(c for c in seq if c in "ACGTN")
    if len(seq) > target:
        seq = seq[:target]
    pad = (6 - len(seq) % 6) % 6
    seq = "A" * pad + seq
    if len(seq) < target:
        seq = "A" * (target - len(seq)) + seq
    return seq

def is_valid(seq: str) -> bool:
    clean = "".join(c for c in seq.upper() if c in "ACGTN")
    if len(clean) < MIN_LEN:
        return False
    if clean.count("N") / max(len(clean), 1) > 0.05:
        return False
    return True

# ── genbank parser ────────────────────────────────────────────────────────────

def extract_from_gb(gb_path: Path) -> list[str]:
    """
    Extract CDS nucleotide sequences from a GenBank file.
    Falls back to the full genomic sequence if no CDS features found.
    Returns up to CDS_CAP sequences per file.
    """
    seqs = []
    try:
        for record in SeqIO.parse(str(gb_path), "genbank"):
            cds_seqs = []
            for feature in record.features:
                if feature.type == "CDS":
                    try:
                        cds_nt = str(feature.extract(record.seq))
                        if is_valid(cds_nt):
                            cds_seqs.append(cds_nt)
                    except Exception:
                        continue
            if cds_seqs:
                seqs.extend(cds_seqs[:CDS_CAP])
            else:
                # No CDS — fall back to full genomic sequence
                full = str(record.seq)
                if is_valid(full):
                    seqs.append(full)
    except Exception as e:
        print(f"  WARN: could not parse {gb_path.name}: {e}")
    return seqs

# ── labelling rules ───────────────────────────────────────────────────────────
#
# Strategy:
#   1. Families with explicit patho/ nopatho/ subdirs → auto-detect
#   2. Top-level families (SARS-CoV-2, MERS-CoV, Coronavirus*) → hardcoded below
#   3. Skip original/ subdirs (already covered by CSV phase 0)
#   4. Skip Coronavirus229E (ambiguous: FIPV/PRCoV in patho/ are animal viruses)

# Relative to BENCH: (family_dir_name, file_stem_or_name) → (family_label, 0/1)
EXPLICIT = {
    # SARS-CoV-2 — the main one is patho, bat relatives are nopatho
    ("SARS-CoV-2", "SARS-CoV-2.gb"): ("SARS-CoV-2", 1),
    ("SARS-CoV-2", "RaTG13.gb"):     ("SARS-CoV-2", 0),
    ("SARS-CoV-2", "Bat-SL-CoVZXC21.gb"): ("SARS-CoV-2", 0),
    ("SARS-CoV-2", "Bat-SL-CoVZC45.gb"):  ("SARS-CoV-2", 0),
    # MERS-CoV — HKU4/HKU5 are bat betacoronaviruses
    ("MERS-CoV", "MERS-CoV.gb"): ("MERS-CoV", 1),
    ("MERS-CoV", "HKU4.gb"):     ("MERS-CoV", 0),
    ("MERS-CoV", "HKU5.gb"):     ("MERS-CoV", 0),
    # CoronavirusNL63 — human pathogen; PEDV/bat = nopatho
    ("CoronavirusNL63", "CoronavirusNL63.gb"): ("CoronavirusNL63", 1),
    ("CoronavirusNL63", "PEDV.gb"):            ("CoronavirusNL63", 0),
    ("CoronavirusNL63", "BtKYNL63-9b.gb"):     ("CoronavirusNL63", 0),
    # CoronavirusOC43 — human pathogen; bovine/canine = nopatho
    ("CoronavirusOC43", "coronavirusOC43.gb"): ("CoronavirusOC43", 1),
    ("CoronavirusOC43", "BCoV.gb"):            ("CoronavirusOC43", 0),
    ("CoronavirusOC43", "CRCoV.gb"):           ("CoronavirusOC43", 0),
    # CoronavirusHKU1 — human pathogen; bovine/mouse = nopatho
    ("CoronavirusHKU1", "coronavirusHKU1.gb"): ("CoronavirusHKU1", 1),
    ("CoronavirusHKU1", "BCoV.gb"):            ("CoronavirusHKU1", 0),
    ("CoronavirusHKU1", "MHV.gb"):             ("CoronavirusHKU1", 0),
}

SKIP_FAMILIES = {"Coronavirus229E"}  # ambiguous: animal viruses in patho/
SKIP_SUBDIRS  = {"original"}         # already covered by CSV

# ── walk JailbreakDNABench ────────────────────────────────────────────────────

gb_sequences, gb_labels, gb_families = [], [], []

for family_dir in sorted(BENCH.iterdir()):
    if not family_dir.is_dir():
        continue
    family = family_dir.name
    if family in SKIP_FAMILIES:
        print(f"  SKIP {family} (ambiguous classification)")
        continue

    # Check for explicit labelling first (top-level .gb files)
    top_level_gb = [f for f in family_dir.glob("*.gb")]
    if top_level_gb:
        for gb_file in sorted(top_level_gb):
            key = (family, gb_file.name)
            if key not in EXPLICIT:
                continue
            fam_label, label = EXPLICIT[key]
            seqs = extract_from_gb(gb_file)
            for s in seqs:
                gb_sequences.append(normalise(s))
                gb_labels.append(label)
                gb_families.append(fam_label if label == 1 else "benign")

    # Auto-detect patho/nopatho subdirs
    for subdir in sorted(family_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in SKIP_SUBDIRS:
            continue
        if subdir.name == "patho":
            label = 1
        elif subdir.name == "nopatho":
            label = 0
        else:
            continue  # skip unknown subdirs (e.g. 数据说明.pdf parent dirs)

        for gb_file in sorted(subdir.glob("*.gb")):
            seqs = extract_from_gb(gb_file)
            for s in seqs:
                gb_sequences.append(normalise(s))
                gb_labels.append(label)
                gb_families.append(family if label == 1 else "benign")

print(f"\nGenBank parsed: {len(gb_sequences)} sequences")
print(f"  Pathogenic: {sum(gb_labels)}")
print(f"  Benign:     {len(gb_labels) - sum(gb_labels)}")

# ── load existing CSV dataset and merge ───────────────────────────────────────

csv_seqs = np.load(PROCESSED / "sequences.npy", allow_pickle=True).tolist()
csv_labs = np.load(PROCESSED / "labels.npy").tolist()
csv_fams = np.load(PROCESSED / "families.npy", allow_pickle=True).tolist()

all_seqs = csv_seqs + gb_sequences
all_labs = csv_labs + gb_labels
all_fams = csv_fams + gb_families
all_srcs = ["csv"] * len(csv_seqs) + ["genbank"] * len(gb_sequences)

# Shuffle
idx = list(range(len(all_seqs)))
random.shuffle(idx)
all_seqs = [all_seqs[i] for i in idx]
all_labs = [all_labs[i] for i in idx]
all_fams = [all_fams[i] for i in idx]
all_srcs = [all_srcs[i] for i in idx]

# ── save ──────────────────────────────────────────────────────────────────────

np.save(PROCESSED / "sequences_full.npy", np.array(all_seqs, dtype=object))
np.save(PROCESSED / "labels_full.npy",    np.array(all_labs, dtype=np.int8))
np.save(PROCESSED / "families_full.npy",  np.array(all_fams, dtype=object))
np.save(PROCESSED / "sources_full.npy",   np.array(all_srcs, dtype=object))

n_patho  = sum(all_labs)
n_benign = len(all_labs) - n_patho
print(f"\nFull dataset saved to {PROCESSED}/")
print(f"  Total:      {len(all_seqs)}")
print(f"  Pathogenic: {n_patho}")
print(f"  Benign:     {n_benign}")

print("\nPathogenic family counts (full):")
for fam, cnt in sorted(Counter(f for f, l in zip(all_fams, all_labs) if l == 1).items()):
    print(f"  {fam:30s} {cnt}")

print("\nGenBank-sourced family breakdown:")
for fam, cnt in sorted(Counter(f for f, l, s in zip(all_fams, all_labs, all_srcs)
                                if s == "genbank" and l == 1).items()):
    print(f"  {fam:30s} {cnt}")
