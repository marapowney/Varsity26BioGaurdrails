#!/usr/bin/env bash
set -euo pipefail

# ── Layer sweep: extract embeddings & PCA-reduce for every layer ──
#
# Usage:
#   ./sweep_layers.sh 7b          # all 32 layers of evo2_7b (balanced dataset)
#   ./sweep_layers.sh 1b          # all 25 layers of evo2_1b_base (balanced dataset)
#   ./sweep_layers.sh 7b 10 20    # sweep layers 10–20 only
#   ./sweep_layers.sh 1b 0 12     # sweep layers 0–12 only

MODEL_SIZE="${1:?Usage: $0 <1b|7b> [start_layer] [end_layer]}"

case "$MODEL_SIZE" in
  7b|7B)
    MODEL_NAME="evo2_7b"
    NUM_LAYERS=32
    DATASET_DIR="probe/data/processed/brca1_v1_128_balanced"
    FEATURES_BASE="probe/features/brca1_v1_128_7b_balanced"
    PCA_BASE="probe/features/brca1_v1_128_7b_balanced_pca95"
    ;;
  1b|1B)
    MODEL_NAME="evo2_1b_base"
    NUM_LAYERS=25
    DATASET_DIR="probe/data/processed/brca1_v1_128_balanced"
    FEATURES_BASE="probe/features/brca1_v1_128_1b_balanced"
    PCA_BASE="probe/features/brca1_v1_128_1b_balanced_pca95"
    ;;
  *)
    echo "Error: unknown model size '$MODEL_SIZE'. Use '1b' or '7b'."
    exit 1
    ;;
esac

START_LAYER="${2:-0}"
END_LAYER="${3:-$((NUM_LAYERS - 1))}"
DEVICE="${DEVICE:-cuda:0}"

echo "============================================================"
echo "  Layer sweep: $MODEL_NAME (layers $START_LAYER–$END_LAYER)"
echo "  Dataset:     $DATASET_DIR"
echo "  Features:    $FEATURES_BASE/blocks.<N>/"
echo "  PCA output:  $PCA_BASE/blocks.<N>/"
echo "  Device:      $DEVICE"
echo "============================================================"
echo ""

# ── Step 1: Extract embeddings for ALL layers (model loaded once) ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Extracting embeddings (layers $START_LAYER–$END_LAYER)"
echo "  The model is loaded once and all layers are extracted per"
echo "  sequence in a single forward pass."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python extract_embeddings.py \
  --model-name "$MODEL_NAME" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$FEATURES_BASE" \
  --layers "$START_LAYER-$END_LAYER" \
  --device "$DEVICE"

# ── Step 2: PCA (95% variance) per layer ──
for LAYER in $(seq "$START_LAYER" "$END_LAYER"); do
  LAYER_NAME="blocks.${LAYER}"
  FEATURES_DIR="$FEATURES_BASE/$LAYER_NAME"
  PCA_DIR="$PCA_BASE/$LAYER_NAME"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  PCA: Layer $LAYER / $END_LAYER  ($LAYER_NAME)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [ -f "$PCA_DIR/embeddings_train.npz" ] && \
     [ -f "$PCA_DIR/embeddings_val.npz" ] && \
     [ -f "$PCA_DIR/embeddings_test.npz" ]; then
    echo "  [pca] Already exists, skipping."
  else
    echo "  [pca] Reducing to 95% variance..."
    python pca_embeddings.py \
      --input-dir "$FEATURES_DIR" \
      --output-dir "$PCA_DIR" \
      --variance-threshold 0.95
  fi

done

echo ""
echo "============================================================"
echo "  Sweep complete!"
echo "  Raw features: $FEATURES_BASE/blocks.{$START_LAYER..$END_LAYER}/"
echo "  PCA features: $PCA_BASE/blocks.{$START_LAYER..$END_LAYER}/"
echo "============================================================"
