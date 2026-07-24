# save as debug_deberta.py in project root and run: python3 debug_deberta.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE    = torch.device("mps")
MODEL     = "microsoft/deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model     = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=5, ignore_mismatched_sizes=True
).to(DEVICE)
model.train()

text   = "What is 2+2? A: 3 B: 4 C: 5 D: 6 E: 7"
enc    = tokenizer(text, return_tensors="pt", max_length=64,
                   padding="max_length", truncation=True).to(DEVICE)
label  = torch.tensor([1], dtype=torch.long).to(DEVICE)

logits = model(**enc).logits
print(f"logits : {logits}")
print(f"any nan: {torch.isnan(logits).any()}")

loss = F.cross_entropy(logits, label)
print(f"loss   : {loss}")