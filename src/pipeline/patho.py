import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

def encode_sequence(sequence, tokenizer, max_length):
    return tokenizer(sequence, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')

PATHO_LM = "../Patho-LM/ckpt"

def patho_check(sequence):
    print("Running PathoLM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n\n\n" + "=" * 10)
    model = AutoModelForSequenceClassification.from_pretrained(PATHO_LM, ignore_mismatched_sizes=True).to(device)
    print("=" * 10 + "\n\n\n")
    tokenizer = AutoTokenizer.from_pretrained(PATHO_LM)
    
    inputs = encode_sequence(sequence, tokenizer, tokenizer.model_max_length)
    with torch.no_grad():
        outputs = model(**{k: v.to(device) for k, v in inputs.items()})
    logits = outputs.logits.cpu().numpy()
    label = np.argmax(logits, axis=1)
    
    if label:
        print(f"Pathogenic score: {np.max(logits, axis=1)}")
    else:
        print(f"Non-pathogenic score: {np.max(logits, axis=1)}")

    return label