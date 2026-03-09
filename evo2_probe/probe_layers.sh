#!/usr/bin/env bash
set -euo pipefail

# ── Layer sweep: train probes on every layer's embeddings ──
#
# Usage:
#   ./probe_layers.sh 7b                        # all 32 layers, linear, raw features
#   ./probe_layers.sh 1b                        # all 25 layers, linear, raw features
#   ./probe_layers.sh 7b --pca                  # linear probe on PCA-reduced features
#   ./probe_layers.sh 7b --mlp                  # MLP probe on raw features
#   ./probe_layers.sh 7b --mlp --pca            # MLP probe on PCA features
#   ./probe_layers.sh 7b 10 20                  # sweep layers 10–20 only
#   ./probe_layers.sh 7b 10 20 --pca --mlp      # layers 10–20, MLP on PCA features

# ── Parse arguments: positional first, then flags ──
MODEL_SIZE="${1:?Usage: $0 <1b|7b|20b> [start_layer] [end_layer] [--pca] [--mlp]}"
shift

START_LAYER=""
END_LAYER=""
USE_PCA=false
USE_MLP=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pca)  USE_PCA=true; shift ;;
    --mlp)  USE_MLP=true; shift ;;
    *)
      if [[ -z "$START_LAYER" ]]; then
        START_LAYER="$1"
      elif [[ -z "$END_LAYER" ]]; then
        END_LAYER="$1"
      else
        echo "Error: unexpected argument '$1'"
        exit 1
      fi
      shift
      ;;
  esac
done

case "$MODEL_SIZE" in
  7b|7B)
    NUM_LAYERS=32
    FEATURES_BASE="probe/features/brca1_v1_128_7b_balanced"
    MODELS_BASE="probe/models/brca1_v1_128_7b_balanced"
    ;;
  1b|1B)
    NUM_LAYERS=25
    FEATURES_BASE="probe/features/brca1_v1_128_1b_balanced"
    MODELS_BASE="probe/models/brca1_v1_128_1b_balanced"
    ;;
  20b|20B)
    NUM_LAYERS=64
    FEATURES_BASE="probe/features/brca1_v1_128_20b_balanced"
    MODELS_BASE="probe/models/brca1_v1_128_20b_balanced"
    ;;
  *)
    echo "Error: unknown model size '$MODEL_SIZE'. Use '1b', '7b', or '20b'."
    exit 1
    ;;
esac

START_LAYER="${START_LAYER:-0}"
END_LAYER="${END_LAYER:-$((NUM_LAYERS - 1))}"

MODEL_TYPE="linear"
if $USE_MLP; then
  MODEL_TYPE="mlp"
  MODELS_BASE="${MODELS_BASE}_mlp"
fi

if $USE_PCA; then
  FEATURES_BASE="${FEATURES_BASE}_pca95"
  MODELS_BASE="${MODELS_BASE}_pca95"
fi

echo "============================================================"
echo "  Probe sweep: $MODEL_SIZE (layers $START_LAYER–$END_LAYER)"
echo "  Model type:  $MODEL_TYPE"
echo "  Features:    $FEATURES_BASE/blocks.<N>/"
echo "  Output:      $MODELS_BASE/blocks.<N>/"
echo "============================================================"
echo ""

FAILED_LAYERS=()

for LAYER in $(seq "$START_LAYER" "$END_LAYER"); do
  LAYER_NAME="blocks.${LAYER}"
  FEATURES_DIR="$FEATURES_BASE/$LAYER_NAME"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Probe: Layer $LAYER / $END_LAYER  ($LAYER_NAME)"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  if [ ! -f "$FEATURES_DIR/embeddings_train.npz" ]; then
    echo "  [skip] Features not found at $FEATURES_DIR, skipping."
    FAILED_LAYERS+=("$LAYER (missing features)")
    continue
  fi

  OUTPUT_DIR="$MODELS_BASE/$LAYER_NAME"
  if [ -f "$OUTPUT_DIR/probe_model.pkl" ]; then
    echo "  [skip] Probe already trained at $OUTPUT_DIR, skipping."
    continue
  fi

  python train_probe.py \
    --features-dir "$FEATURES_BASE" \
    --output-dir "$MODELS_BASE" \
    --layer "$LAYER" \
    --model "$MODEL_TYPE" \
    || { echo "  [FAIL] Layer $LAYER failed."; FAILED_LAYERS+=("$LAYER"); }

done

echo ""
echo "============================================================"
echo "  Probe sweep complete!"
echo "  Model type:  $MODEL_TYPE"
echo "  Features:    $FEATURES_BASE/blocks.{$START_LAYER..$END_LAYER}/"
echo "  Outputs:     $MODELS_BASE/blocks.{$START_LAYER..$END_LAYER}/"
if [ ${#FAILED_LAYERS[@]} -gt 0 ]; then
  echo "  Skipped/failed layers: ${FAILED_LAYERS[*]}"
fi
echo "============================================================"
