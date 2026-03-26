"""
train_sentiment.py
Fine-tune CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment on the TUNIZI dataset.
Architecture: AutoModelForSequenceClassification + Trainer (fully explicit, no pipeline shortcut).
"""

import sys
import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from torch.utils.data import Dataset

# ─── CUDA GUARD ───────────────────────────────────────────────────────────────
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Refusing to start — GPU required for this training run.")
    sys.exit(1)
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("C:/Users/moham/GOMYCODE/Insurance-AI-Chatbot")
DATA_PATH  = BASE_DIR / "TUNIZI-Sentiment-Analysis-Tunisian-Arabizi-Dataset-master" / "TUNIZI-Dataset.txt"
MODEL_DIR  = BASE_DIR / "ml_models" / "sentiment_model"
MODEL_NAME = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"

# ─── STEP 1: LOAD & PARSE ──────────────────────────────────────────────────────
print("\n=== STEP 1: Loading & Parsing TUNIZI Dataset ===")
df = pd.read_csv(DATA_PATH, sep=";", header=None, names=["label", "text"], encoding="utf-8")

# Strip stray % chars from labels (some lines start with %)
df["label"] = df["label"].astype(str).str.replace("%", "", regex=False)
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Map: 1 → 0 (calm), -1 → 1 (urgent/angry)
df["label"] = df["label"].map({1: 0, -1: 1})
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print(f"Total valid samples: {len(df)}")
print("Class distribution:")
print(df["label"].value_counts().rename({0: "calm (0)", 1: "urgent (1)"}))

# ─── STEP 2: TOKENIZER & SPLITS ───────────────────────────────────────────────
print(f"\n=== STEP 2: Fine-tuning {MODEL_NAME} ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

texts  = df["text"].tolist()
labels = df["label"].tolist()

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


train_dataset = SentimentDataset(X_train, y_train, tokenizer)
val_dataset   = SentimentDataset(X_val,   y_val,   tokenizer)
test_dataset  = SentimentDataset(X_test,  y_test,  tokenizer)

# ─── MODEL ────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True,   # CaMEL has 3-label head; we override to 2
)

# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels, preds, average="binary"),
    }

# ─── TRAINING ARGS ────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "checkpoints"),
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_dir=str(MODEL_DIR / "logs"),
    logging_steps=50,
    fp16=True,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()

# ─── STEP 3: EVALUATE & SAVE ──────────────────────────────────────────────────
print("\n=== STEP 3: Test Split Evaluation ===")
preds_output = trainer.predict(test_dataset)
y_pred = np.argmax(preds_output.predictions, axis=-1)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["calm", "urgent"]))

# Save model + tokenizer
MODEL_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))

# Save label map
label_map = {"0": "calm", "1": "urgent"}
with open(MODEL_DIR / "label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\nModel saved to {MODEL_DIR}")
print("Training complete.")
