"""
Train BERTweet with class weights and label smoothing for sentiment analysis.
"""

import os
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from utils import (
    set_seeds,
    load_tweet_eval_dataset,
    compute_metrics,
    ID2LABEL,
    LABEL2ID,
    LABEL_MAP,
    evaluate_and_visualize,
)


class WeightedSmoothedTrainer(Trainer):
    """Custom Trainer with class weights and label smoothing."""
    
    def __init__(self, class_weights=None, label_smoothing=0.05, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with class weights and label smoothing."""
        labels = inputs["labels"]
        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        
        outputs = model(**model_inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(
                weight=weights,
                label_smoothing=self.label_smoothing,
            )
        else:
            loss_fct = nn.CrossEntropyLoss(
                label_smoothing=self.label_smoothing,
            )
        
        loss = loss_fct(
            logits.view(-1, model.config.num_labels),
            labels.view(-1),
        )
        
        return (loss, outputs) if return_outputs else loss


def main():
    """Train BERTweet with class weights and label smoothing."""
    # Set seeds
    set_seeds(42)
    
    # Set wandb environment
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load dataset
    print("Loading dataset...")
    ds = load_tweet_eval_dataset()
    
    # Compute class weights from training data
    train_labels = ds["train"]["label"]
    label_counts = Counter(train_labels)
    print("Label counts:", label_counts)
    
    num_labels = len(LABEL_MAP)
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / label_counts[i] for i in range(num_labels)],
        dtype=torch.float,
    )
    
    # Normalize so that mean weight is 1
    class_weights = class_weights / class_weights.mean()
    print("class_weights:", class_weights)
    
    # Tokenizer and encoding
    bertweet_model_name = "vinai/bertweet-base"
    
    bertweet_tokenizer = AutoTokenizer.from_pretrained(
        bertweet_model_name,
        normalization=True
    )
    
    max_length = 128
    
    def tokenize_bertweet(examples):
        return bertweet_tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )
    
    bertweet_encoded_ds = ds.map(tokenize_bertweet, batched=True)
    bertweet_encoded_ds = bertweet_encoded_ds.remove_columns(["text"])
    bertweet_encoded_ds = bertweet_encoded_ds.rename_column("label", "labels")
    bertweet_encoded_ds.set_format("torch")
    
    bertweet_data_collator = DataCollatorWithPadding(tokenizer=bertweet_tokenizer)
    
    # Load model
    print(f"\nLoading {bertweet_model_name}...")
    bertweet_ws_model = AutoModelForSequenceClassification.from_pretrained(
        bertweet_model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Training arguments
    bertweet_ws_training_args = TrainingArguments(
        output_dir="./bertweet-sentiment-ws",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
    )
    
    # Trainer with class weights and label smoothing
    bertweet_ws_trainer = WeightedSmoothedTrainer(
        model=bertweet_ws_model,
        args=bertweet_ws_training_args,
        train_dataset=bertweet_encoded_ds["train"],
        eval_dataset=bertweet_encoded_ds["validation"],
        tokenizer=bertweet_tokenizer,
        data_collator=bertweet_data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        label_smoothing=0.05,
    )
    
    # Train
    print("\nStarting training...")
    bertweet_ws_trainer.train()
    print("\nWeighted + smoothed BERTweet training finished.")
    
    # Evaluation
    bertweet_ws_val_results = bertweet_ws_trainer.evaluate(bertweet_encoded_ds["validation"])
    print("\n=== Val metrics (BERTweet + weights + label smoothing) ===")
    for k, v in bertweet_ws_val_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    bertweet_ws_test_results = bertweet_ws_trainer.evaluate(bertweet_encoded_ds["test"])
    print("\n=== Test metrics (BERTweet + weights + label smoothing) ===")
    for k, v in bertweet_ws_test_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    # Detailed evaluation and visualization
    print("\n=== Generating visualizations ===")
    evaluate_and_visualize(
        bertweet_ws_trainer,
        bertweet_encoded_ds["test"],
        "BERTweet (weights + smoothing)",
        save_dir="./all_png/bertweet_weighted"
    )


if __name__ == "__main__":
    main()


