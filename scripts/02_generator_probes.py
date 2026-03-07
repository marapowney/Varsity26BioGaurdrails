"""
Phase 2 — GENERator activation space probes (SG2.1–2.5)
+ Phase 3 — ROC comparison vs k-mer baseline (SG3.1–3.2)

Extracts mean-pooled hidden states from GENERator-v2-eukaryote-1.2b-base at
4 intermediate layers, trains a logistic regression probe per layer, reports
cross-validated AUC, and saves the best probe + embeddings for downstream use.

Output:
  data/embeddings/<layer_name>.npy   (84, hidden_dim) float32 per layer
  data/probes/best_probe.pkl         {probe, scaler, layer_name, layer_idx}
  data/processed/probe_results.npz   scores + metadata
  plots/roc_comparison.png           k-mer vs probe ROC curves
  plots/probe_per_layer.png          AUC per layer bar chart
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT      = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EMB_DIR   = ROOT / "data" / "embeddings"
PROBE_DIR = ROOT / "data" / "probes"
PLOTS     = ROOT / "plots"

EMB_DIR.mkdir(parents=True, exist_ok=True)
PROBE_DIR.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(exist_ok=True)

# Two options — swap and rerun to compare:
#   eukaryote: trained on gene-centric RefSeq (GCP), CDS-biased → better for coding viral genes
#   prokaryote: trained on prokaryotic RefSeq, more viral/phage exposure → may separate pathogens better
# MODEL_NAME = "GenerTeam/GENERator-v2-prokaryote-1.2b-base"
MODEL_NAME = "GenerTeam/GENERator-v2-eukaryote-1.2b-base"

# ── load data ────────────────────────────────────────────────────────────────

sequences = np.load(PROCESSED / "sequences.npy", allow_pickle=True).tolist()
labels    = np.load(PROCESSED / "labels.npy").tolist()
families  = np.load(PROCESSED / "families.npy", allow_pickle=True).tolist()
labels_arr = np.array(labels)
print(f"Loaded {len(sequences)} sequences ({sum(labels)} patho, {labels_arr.sum() - sum(labels) + (labels_arr == 0).sum()} benign)")

# ── load model (SG2.1) ────────────────────────────────────────────────────────

print(f"\nLoading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()

n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Model: {n_layers} transformer layers, hidden_dim={hidden_dim}")
assert tokenizer.vocab_size >= 4096, "Expected 6-mer tokenizer (4^6=4096 vocab)"
print("Tokenizer vocab size:", tokenizer.vocab_size, "✓")

LAYERS = {
    "early": n_layers // 4,
    "mid":   n_layers // 2,
    "late":  3 * n_layers // 4,
    "last":  n_layers,       # index into hidden_states tuple (0=embed, n=final)
}
print(f"Probing layers: { {k: v for k, v in LAYERS.items()} }")

# ── embedding extraction (SG2.2–2.3) ─────────────────────────────────────────

def get_embedding(seq: str, layer_idx: int) -> np.ndarray:
    """Mean-pooled hidden state at layer_idx (index into hidden_states tuple)."""
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
    hidden = outputs.hidden_states[layer_idx]           # (1, seq_len, hidden_dim)
    mask   = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)   # (1, hidden_dim)
    return pooled.squeeze(0).cpu().float().numpy()

# Check if embeddings already saved — skip extraction if so
missing = [name for name in LAYERS if not (EMB_DIR / f"{name}.npy").exists()]

if missing:
    print(f"\nExtracting embeddings for layers: {missing}")
    for layer_name in missing:
        layer_idx = LAYERS[layer_name]
        print(f"  Layer '{layer_name}' (hidden_states[{layer_idx}])...")
        embs = [get_embedding(seq, layer_idx) for seq in tqdm(sequences, desc=layer_name)]
        arr = np.stack(embs).astype(np.float32)
        np.save(EMB_DIR / f"{layer_name}.npy", arr)
        print(f"  Saved {arr.shape} → data/embeddings/{layer_name}.npy")
else:
    print(f"\nAll embeddings already extracted — loading from disk.")

embeddings = {name: np.load(EMB_DIR / f"{name}.npy") for name in LAYERS}

# ── probe training (SG2.4–2.5) ───────────────────────────────────────────────

print("\nTraining probes (StratifiedKFold 5, Pipeline with scaler)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

probe_results = {}
for layer_name, X in embeddings.items():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)),
    ])
    cv_out = cross_validate(pipe, X, labels_arr, cv=cv,
                            scoring=["roc_auc", "f1"], return_train_score=False)
    auc_mean = cv_out["test_roc_auc"].mean()
    auc_std  = cv_out["test_roc_auc"].std()
    f1_mean  = cv_out["test_f1"].mean()

    # Fit on full dataset for deployment
    pipe.fit(X, labels_arr)

    probe_results[layer_name] = {
        "auc":   auc_mean,
        "std":   auc_std,
        "f1":    f1_mean,
        "pipe":  pipe,
    }
    print(f"  [{layer_name:5s}] AUC = {auc_mean:.3f} ± {auc_std:.3f}  F1 = {f1_mean:.3f}")

best_layer = max(probe_results, key=lambda k: probe_results[k]["auc"])
print(f"\nBest layer: {best_layer}  (AUC={probe_results[best_layer]['auc']:.3f})")

# Save best probe
best_probe_path = PROBE_DIR / "best_probe.pkl"
with open(best_probe_path, "wb") as f:
    pickle.dump({
        "pipe":       probe_results[best_layer]["pipe"],
        "layer_name": best_layer,
        "layer_idx":  LAYERS[best_layer],
    }, f)
print(f"Best probe saved → {best_probe_path}")

# ── per-family stratified AUC (SG3.2) ────────────────────────────────────────

print("\nPer-family AUC (pathogenic family vs all 40 benign):")
best_X = embeddings[best_layer]
best_pipe = probe_results[best_layer]["pipe"]
probe_scores = best_pipe.predict_proba(best_X)[:, 1]

benign_mask  = labels_arr == 0
benign_probe = probe_scores[benign_mask]

patho_families = sorted(set(f for f, l in zip(families, labels) if l == 1))
for fam in patho_families:
    fam_mask = np.array([f == fam for f in families])
    combined_labels = np.array([1] * fam_mask.sum() + [0] * benign_mask.sum())
    combined_probe  = np.concatenate([probe_scores[fam_mask], benign_probe])
    auc = roc_auc_score(combined_labels, combined_probe)
    print(f"  {fam:12s}  probe AUC={auc:.3f}  (n={fam_mask.sum()})")

# ── save probe scores ─────────────────────────────────────────────────────────

np.savez(
    PROCESSED / "probe_results.npz",
    probe_scores=probe_scores,
    labels=labels_arr,
    families=np.array(families, dtype=object),
    best_layer=best_layer,
)
print("\nProbe scores saved → data/processed/probe_results.npz")

# ── ROC comparison plot (SG3.1) ───────────────────────────────────────────────

kmer_data   = np.load(PROCESSED / "kmer_results.npz", allow_pickle=True)
kmer_scores = kmer_data["top_scores"]   # max per-family (best kmer variant)

fig, ax = plt.subplots(figsize=(7, 6))
for scores, name, color in [
    (kmer_scores,  f"K-mer max-family (AUC={roc_auc_score(labels_arr, kmer_scores):.3f})", "steelblue"),
    (probe_scores, f"GENERator probe [{best_layer}] (AUC={roc_auc_score(labels_arr, probe_scores):.3f})", "crimson"),
]:
    fpr, tpr, _ = roc_curve(labels_arr, scores)
    ax.plot(fpr, tpr, lw=2, color=color, label=name)

ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("K-mer vs GENERator Probe — Pathogenicity Detection ROC")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "roc_comparison.png", dpi=150)
print(f"ROC comparison saved → plots/roc_comparison.png")

# ── per-layer AUC bar chart ───────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
layer_names = list(probe_results.keys())
aucs  = [probe_results[n]["auc"] for n in layer_names]
stds  = [probe_results[n]["std"] for n in layer_names]
colors = ["crimson" if n == best_layer else "steelblue" for n in layer_names]
bars = ax.bar(layer_names, aucs, yerr=stds, color=colors, capsize=4, alpha=0.8)
ax.axhline(roc_auc_score(labels_arr, kmer_scores), color="orange",
           linestyle="--", lw=1.5, label=f"K-mer baseline ({roc_auc_score(labels_arr, kmer_scores):.3f})")
ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="Random (0.500)")
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("CV AUC-ROC (5-fold)")
ax.set_title("GENERator Probe AUC by Layer")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "probe_per_layer.png", dpi=150)
print(f"Per-layer AUC chart saved → plots/probe_per_layer.png")
