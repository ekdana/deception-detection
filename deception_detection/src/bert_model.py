# src/bert_model.py

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# =====================================================
# 1. DATASET CLASS (same as your implementation)
# =====================================================

class ReviewDataset(Dataset):
    """
    Custom PyTorch Dataset for BERT training.
    Converts review texts into tokenized inputs.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# =====================================================
# 2. METRICS FUNCTION (same logic as your code)
# =====================================================

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and F1 score.
    """

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary"
    )

    acc = accuracy_score(labels, predictions)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# =====================================================
# 3. MAIN TRAINING FUNCTION
# =====================================================

def train_bert_model(train_texts, test_texts, train_labels, test_labels):
    """
    Fine-tunes BERT-base model for deceptive review classification.
    Based directly on your project pipeline.
    """

    print("\n==============================")
    print("   BERT Training Started")
    print("==============================")

    model_name = "bert-base-uncased"

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # Create datasets
    train_dataset = ReviewDataset(
        train_texts,
        train_labels,
        tokenizer
    )

    test_dataset = ReviewDataset(
        test_texts,
        test_labels,
        tokenizer
    )

    # Training configuration
    training_args = TrainingArguments(
        output_dir="./bert_results",

        # IMPORTANT FIX: correct parameter name
        eval_strategy="epoch",

        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,

        logging_dir="./logs",
        logging_steps=20
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()

    print("\n==============================")
    print("   Final Evaluation Results")
    print("==============================")
    print(results)

    return model, tokenizer, results
