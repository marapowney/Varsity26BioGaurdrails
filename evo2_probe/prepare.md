# Extract Evo2 Embeddings

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

### Evo2 1B Base

25 layers total (blocks.0–24). Layer 18 (~75% depth) is recommended — the
final layers tend to over-specialize for next-token prediction, so a layer
around the 3/4 mark usually gives better representations for linear probing.

```bash
python extract_embeddings.py \
  --model-name evo2_1b_base \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_1b_balanced \
  --layer 18 \
  --device cuda:0
```

### Evo2 7B

32 layers total (blocks.0–31). Layer 23 (~75% depth).

```bash
python extract_embeddings.py \
  --model-name evo2_7b \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_7b_balanced \
  --layer 23 \
  --device cuda:0
```

### Evo2 7B Base

32 layers total (blocks.0–31). Same architecture as 7B but without
instruction tuning.

```bash
python extract_embeddings.py \
  --model-name evo2_7b_base \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_7b_base_balanced \
  --layer 23 \
  --device cuda:0
```

### Evo2 7B 262K

32 layers total (blocks.0–31). Long-context (262k) variant of Evo2 7B.

```bash
python extract_embeddings.py \
  --model-name evo2_7b_262k \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_7b_262k_balanced \
  --layer 23 \
  --device cuda:0
```

### Evo2 7B Microviridae

32 layers total (blocks.0–31). Fine-tuned on microviridae genomes.

```bash
python extract_embeddings.py \
  --model-name evo2_7b_microviridae \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_7b_microviridae_balanced \
  --layer 23 \
  --device cuda:0
```

### Evo2 20B

24 layers total (blocks.0–23). Layer 17 (~75% depth).

```bash
python extract_embeddings.py \
  --model-name evo2_20b \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_20b_balanced \
  --layer 17 \
  --device cuda:0
```

### Evo2 40B

50 layers total (blocks.0–49). Layer 37 (~75% depth).

```bash
python extract_embeddings.py \
  --model-name evo2_40b \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_40b_balanced \
  --layer 37 \
  --device cuda:0
```

### Evo2 40B Base

50 layers total (blocks.0–49). Same architecture as 40B but without
instruction tuning.

```bash
python extract_embeddings.py \
  --model-name evo2_40b_base \
  --dataset-dir probe/data/processed/brca1_v1_128_balanced \
  --output-dir probe/features/brca1_v1_128_40b_base_balanced \
  --layer 37 \
  --device cuda:0
```

## Unbalanced dataset

### Evo2 1B Base

```bash
python extract_embeddings.py \
  --model-name evo2_1b_base \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_1b \
  --layer 18 \
  --device cuda:0
```

### Evo2 7B

```bash
python extract_embeddings.py \
  --model-name evo2_7b \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_7b \
  --layer 23 \
  --device cuda:0
```

### Evo2 7B Base

```bash
python extract_embeddings.py \
  --model-name evo2_7b_base \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_7b_base \
  --layer 23 \
  --device cuda:0
```

### Evo2 7B 262K

```bash
python extract_embeddings.py \
  --model-name evo2_7b_262k \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_7b_262k \
  --layer 23 \
  --device cuda:0
```

### Evo2 7B Microviridae

```bash
python extract_embeddings.py \
  --model-name evo2_7b_microviridae \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_7b_microviridae \
  --layer 23 \
  --device cuda:0
```

### Evo2 20B

```bash
python extract_embeddings.py \
  --model-name evo2_20b \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_20b \
  --layer 17 \
  --device cuda:0
```

### Evo2 40B

```bash
python extract_embeddings.py \
  --model-name evo2_40b \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_40b \
  --layer 37 \
  --device cuda:0
```

### Evo2 40B Base

```bash
python extract_embeddings.py \
  --model-name evo2_40b_base \
  --dataset-dir probe/data/processed/brca1_v1_128 \
  --output-dir probe/features/brca1_v1_128_40b_base \
  --layer 37 \
  --device cuda:0
```

## Useful options

- `--layer <N>` — extract from a specific layer (e.g. `--layer 23`). Defaults to the last block.
- `--list-layers` — print all valid layer names for the model and exit.
- `--no-flash-attn` — fall back to PyTorch SDPA if flash-attn is not installed.
- `--checkpoint-every <N>` — save progress every N unique sequences (default 200).
