
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

LABELS = [
    "ATM / Debit Cards", "Credit Cards", "Deposit Accounts",
    "Electronic Banking / Mobile", "Loans and Advances",
    "Others", "Recovery Agents", "Remittance / Money Transfer"
]

MODEL_PATH = "../models/distilbert-rbi-complaint"

def predict(text):
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred  = torch.argmax(probs).item()
    return LABELS[pred], round(probs[pred].item() * 100, 2)

# Test with 3 Indian banking complaints
tests = [
    "My ATM card was blocked after 3 wrong PIN attempts. I cannot withdraw cash.",
    "The bank has deducted EMI twice this month for my home loan account.",
    "I sent money via NEFT to wrong account. Bank is not helping me recover it."
]

for t in tests:
    label, conf = predict(t)
    print(f"Complaint: {t[:60]}...")
    print(f"Category:  {label} ({conf}% confidence)")
    print()
