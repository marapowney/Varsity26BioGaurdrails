# Training the Probe

## Evo2 7B

### Linear (logistic regression)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_7b \
  --output-dir probe/models/brca1_v1_128_7b \
  --feature-type mean_pool
```

### Non-linear (MLP)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_7b \
  --output-dir probe/models/brca1_v1_128_7b_mlp \
  --feature-type mean_pool \
  --model mlp
```

## Evo2 1B Base

### Linear (logistic regression)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_1b \
  --output-dir probe/models/brca1_v1_128_1b \
  --feature-type mean_pool
```

### Non-linear (MLP)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_1b \
  --output-dir probe/models/brca1_v1_128_1b_mlp \
  --feature-type mean_pool \
  --model mlp
```

## Balanced dataset — Evo2 7B

### Linear (logistic regression)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_7b_balanced \
  --output-dir probe/models/brca1_v1_128_7b_balanced \
  --feature-type mean_pool
```

### Non-linear (MLP)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_7b_balanced \
  --output-dir probe/models/brca1_v1_128_7b_balanced_mlp \
  --feature-type mean_pool \
  --model mlp
```

## Balanced dataset — Evo2 1B Base

### Linear (logistic regression)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_1b_balanced \
  --output-dir probe/models/brca1_v1_128_1b_balanced \
  --feature-type mean_pool
```

### Non-linear (MLP)

```bash
python train_probe.py \
  --features-dir probe/features/brca1_v1_128_1b_balanced \
  --output-dir probe/models/brca1_v1_128_1b_balanced_mlp \
  --feature-type mean_pool \
  --model mlp
```
