# Linear Probe Plan: Safe vs Unsafe Human DNA Subsequence Classification with Evo2

## 1) Objective

Build a reproducible, leakage-resistant linear-probe pipeline on frozen Evo2 embeddings to classify human DNA subsequences into:

- `safe`: normal/reference context and benign/likely-benign variant contexts.
- `unsafe`: pathogenic/likely-pathogenic or strong loss-of-function (LOF) variant contexts.

The immediate target is a BRCA1-focused proof of concept using data already present in this repo, followed by expansion to a broader human variant dataset.

## 2) Scope and Definitions

### In scope

1. Binary classification on fixed-length human DNA windows.
2. Frozen Evo2 embeddings as features.
3. Linear classifiers only (for interpretability and simplicity).
4. Strictly offline evaluation and model cards.

### Out of scope

1. End-to-end fine-tuning of Evo2.
2. Clinical-grade interpretation or decision support.
3. Any sequence-generation workflow.

### Label definition

1. `safe`
   - Reference/normal windows (no known pathogenic variant injected).
   - Benign and likely benign variant windows.
2. `unsafe`
   - Pathogenic and likely pathogenic variant windows.
   - Variants with clear experimental LOF evidence.
3. `exclude`
   - VUS/uncertain/conflicting interpretation.
   - Inconsistent or low-confidence labels.

## 3) Available Starting Assets in This Repo

1. BRCA1 variant dataset:
   - `notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx`
   - Contains `func.class`, `clinvar`, and per-variant metadata.
2. Human reference sequence:
   - `notebooks/brca1/GRCh37.p13_chr17.fna.gz`
3. Existing BRCA1 notebook logic:
   - `notebooks/brca1/brca1_zero_shot_vep.ipynb`
   - Includes variant normalization and local sequence extraction workflow.

## 4) End-to-End Workplan

## Phase A: Project Setup and Reproducibility

1. Create a new module directory structure:
   - `probe/`
   - `probe/config/`
   - `probe/data/`
   - `probe/features/`
   - `probe/models/`
   - `probe/eval/`
2. Add deterministic run settings:
   - Fixed seeds.
   - Logged package versions.
   - Config-driven runs with YAML/JSON.
3. Add experiment tracker outputs:
   - Input config snapshot.
   - Hashes for datasets and split files.
   - Metrics JSON and confusion matrices.

Deliverables:

1. Reproducible run harness.
2. Skeleton CLI scripts and configs.

## Phase B: Label Policy and Data Spec

1. Build a label mapping table from source fields:
   - `clinvar`: map `benign/likely benign` to `safe`; `pathogenic/likely pathogenic` to `unsafe`.
   - `func.class`: map `LOF` to `unsafe`; map `FUNC` to `safe`; treat `INT` as policy-based (default: exclude for strictness in v1).
2. Resolve conflicts between fields with explicit precedence:
   - Rule example: if any strong pathogenic evidence exists, label `unsafe`.
   - Otherwise safe only if all available labels are benign/functional.
3. Document all inclusion/exclusion logic in a versioned YAML policy file.

Deliverables:

1. `label_policy_v1.yaml`
2. `label_mapping_report.md` with class counts.

## Phase C: Dataset Construction

1. Normalize variant representation:
   - Left-normalize and standardize REF/ALT.
   - Validate against reference base at coordinate.
2. Generate windows per example:
   - Candidate lengths: 512, 1024, 2048, 4096, 8192.
   - Center variant where possible.
   - For reference-only safe examples, sample matched windows in same loci.
3. Build example types:
   - `ref_window` at variant locus.
   - `alt_window` with variant injected.
   - Optional pairwise feature support (`alt - ref`).
4. Quality checks:
   - Ensure valid DNA alphabet.
   - No duplicate identical sequence windows across rows.
   - Log dropped rows and reasons.

Deliverables:

1. `dataset_v1.parquet` (or sharded parquet).
2. `dataset_qc_report.md`.

## Phase D: Split Strategy to Prevent Leakage

1. Use split units that prevent near-duplicate overlap:
   - Group by genomic locus bins or variant position clusters.
   - Optionally group by exon/domain to force harder generalization.
2. Recommended split proportions:
   - Train 70%, validation 15%, test 15%.
3. Create a second, harder out-of-region test split:
   - Held-out contiguous locus ranges.
4. Enforce split reproducibility with persisted split files.

Deliverables:

1. `splits_v1.jsonl`
2. `split_leakage_audit.md`

## Phase E: Frozen Evo2 Feature Extraction

1. Start with one lightweight checkpoint for fast iteration:
   - `evo2_1b_base` first.
2. Later replicate with stronger checkpoints:
   - `evo2_7b` and optionally larger models if available.
3. Feature families:
   - Mean pooled token embeddings.
   - Final token embedding.
   - Multi-layer concatenation or single selected layer.
   - Variant-delta representation: `h_alt - h_ref`.
4. Persist feature caches keyed by:
   - model name, layer set, window size, tokenizer version, split version.

Deliverables:

1. `features_<model>_<window>_<layer>.npy/parquet`
2. `feature_manifest.json`

## Phase F: Linear Probe Training

1. Baseline classifier:
   - Logistic regression with L2 regularization.
2. Optional linear baselines:
   - Linear SVM.
   - Ridge classifier.
3. Imbalance handling:
   - Class weighting.
   - Optional controlled downsampling of majority class.
4. Hyperparameter search:
   - Regularization strength.
   - Window size.
   - Layer choice.
   - Feature type (`alt`, `ref`, `delta`, concatenated).
5. Calibration:
   - Platt scaling or isotonic on validation set only.

Deliverables:

1. `probe_model_v1.pkl`
2. `train_summary_v1.md`

## Phase G: Evaluation and Error Analysis

1. Primary metrics:
   - AUROC, AUPRC, F1, precision, recall.
2. Thresholded performance:
   - Recall at high precision targets.
   - Precision at high recall targets.
3. Subgroup metrics:
   - By consequence type (missense, nonsense, splice).
   - By allele frequency bins (if available).
   - By genomic region buckets.
4. Robustness checks:
   - Sequence reverse-complement consistency (if expected).
   - Sensitivity to small context shifts.
5. Error inspection:
   - Top false positives and false negatives with metadata.
   - Label-noise review candidates.

Deliverables:

1. `eval_report_v1.md`
2. `metrics_v1.json`
3. `error_cases_v1.csv`

## Phase H: Packaging and Inference Interface

1. Provide a strict inference API:
   - Input: sequence window or variant + locus + reference context.
   - Output: unsafe probability, calibrated confidence, class label.
2. Include metadata in every prediction:
   - model version, feature config, threshold version.
3. Include guardrails in docs:
   - Research-use-only disclaimer.
   - Not a clinical diagnostic system.

Deliverables:

1. `probe/predict.py`
2. `MODEL_CARD.md`

## Phase I: Expansion Beyond BRCA1

1. Add external human variant sources (phase-gated):
   - ClinVar-derived pathogenic/benign labels.
   - Additional genes with high-quality curated labels.
2. Keep BRCA1 entirely held out in one experiment to test generalization.
3. Retrain same linear-probe framework with no architecture changes.

Deliverables:

1. `dataset_v2` (multi-gene).
2. Cross-gene generalization report.

## 5) Concrete Script Plan

1. `probe/data/build_dataset.py`
   - Load source variants, apply label policy, generate windows.
2. `probe/data/make_splits.py`
   - Group-aware split generation and leakage checks.
3. `probe/features/extract_embeddings.py`
   - Frozen Evo2 embedding extraction and caching.
4. `probe/models/train_linear_probe.py`
   - Train logistic/linear models and save artifacts.
5. `probe/eval/evaluate_probe.py`
   - Full metric suite and subgroup reports.
6. `probe/predict.py`
   - Batch and single-example inference entrypoint.

## 6) Success Criteria

1. Reproducibility:
   - Re-running with same config reproduces metrics within tight tolerance.
2. Performance:
   - Strong discrimination on held-out test (target AUROC > 0.80 for v1 BRCA1 PoC).
3. Robustness:
   - No major leakage indicators.
   - Stable performance across subgroup slices.
4. Operability:
   - One-command train/eval flow with persisted artifacts.

## 7) Risk Register and Mitigation

1. Label noise and contradictions.
   - Mitigation: strict policy, exclusions, conflict report.
2. Train/test leakage from overlapping genomic neighborhoods.
   - Mitigation: group-aware splits and explicit leakage audit.
3. Class imbalance causing unstable thresholds.
   - Mitigation: class weighting + calibration + PR-oriented selection.
4. Overfitting to BRCA1 context.
   - Mitigation: out-of-region tests and multi-gene phase.
5. Confounding by sequence composition artifacts.
   - Mitigation: matched safe controls and stratified evaluation.

## 8) Compute and Runtime Expectations

1. PoC training (linear models) is lightweight once embeddings are cached.
2. Embedding extraction is dominant cost.
3. Start with smaller Evo2 checkpoint and shorter windows for iteration speed.
4. Scale to larger checkpoints/windows after split/label pipeline stabilizes.

## 9) Suggested Milestones

1. Milestone 1: Data and labels finalized.
   - Output: `dataset_v1`, label report, leakage-safe splits.
2. Milestone 2: Embedding extraction complete.
   - Output: cached feature store for all splits.
3. Milestone 3: Baseline probe + calibrated threshold.
   - Output: model artifact, validation curves.
4. Milestone 4: Full evaluation and error analysis.
   - Output: test report and model card draft.
5. Milestone 5: Multi-gene expansion plan executed.
   - Output: `dataset_v2` and cross-gene generalization metrics.

## 10) Immediate Next Actions

1. Freeze `label_policy_v1`.
2. Implement `build_dataset.py` for BRCA1-only PoC.
3. Implement group-aware split generator.
4. Extract first embedding set with `evo2_1b_base`.
5. Train first logistic probe and generate calibration/eval reports.
