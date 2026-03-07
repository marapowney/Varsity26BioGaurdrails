# Extract Evo2 Embeddings

## Evo2 7B

32 layers total (blocks.0–31). Layer 23 (~75% depth) is recommended — the
final layers tend to over-specialize for next-token prediction, so a layer
around the 3/4 mark usually gives better representations for linear probing.

```bash
python extract_embeddings.py \
  --model-name evo2_7b \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_7b \
  --layer 23 \
  --device cuda:0
```

## Evo2 1B Base

25 layers total (blocks.0–24). Same heuristic puts the sweet spot around
layer 18.

```bash
python extract_embeddings.py \
  --model-name evo2_1b_base \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_1b \
  --layer 18 \
  --device cuda:0
```

## Balanced dataset

Build a balanced raw dataset first (downsamples safe to match unsafe per
split), then extract embeddings from it — avoids wasting GPU on discarded
sequences.

```bash
python build_dataset.py \
  --window-size 128 \
  --output-dir probe/data/processed/brca1_v1_128_balanced \
  --balance
```

### Evo2 7B

```bash
python extract_embeddings.py \
  --model-name evo2_7b \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_7b_balanced \
  --layer 23 \
  --device cuda:0
```

### Evo2 1B Base

```bash
python extract_embeddings.py \
  --model-name evo2_1b_base \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_1b_balanced \
  --layer 18 \
  --device cuda:0
```

## Useful options

- `--layer <N>` — extract from a specific layer (e.g. `--layer 23`). Defaults to the last block.
- `--list-layers` — print all valid layer names for the model and exit.
- `--no-flash-attn` — fall back to PyTorch SDPA if flash-attn is not installed.
- `--checkpoint-every <N>` — save progress every N unique sequences (default 200).
