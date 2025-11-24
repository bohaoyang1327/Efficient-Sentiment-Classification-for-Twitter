"""
Train RoBERTa-base model for sentiment analysis.
"""

import os
import numpy as np
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
    evaluate_and_visualize,
)


def main():
    """Train RoBERTa-base model."""
    # Set seeds
    set_seeds(42)
    
    # Set wandb environment
    os.environ["WANDB_PROJECT"] = "tweet-eval-sentiment"
    os.environ["WANDB_WATCH"] = "false"
    
    # Load dataset
    print("Loading dataset...")
    ds = load_tweet_eval_dataset()
    
    # Tokenizer and preprocessing
    roberta_model_name = "roberta-base"
    roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
    
    def preprocess_roberta(examples):
        return roberta_tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )
    
    # Re-tokenize dataset for RoBERTa
    roberta_encoded_ds = ds.map(preprocess_roberta, batched=True)
    roberta_encoded_ds = roberta_encoded_ds.remove_columns(["text"])
    roberta_encoded_ds = roberta_encoded_ds.rename_column("label", "labels")
    roberta_encoded_ds.set_format("torch")
    
    roberta_data_collator = DataCollatorWithPadding(tokenizer=roberta_tokenizer)
    
    # Load RoBERTa model
    print(f"\nLoading {roberta_model_name}...")
    roberta_model = AutoModelForSequenceClassification.from_pretrained(
        roberta_model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Training arguments
    roberta_training_args = TrainingArguments(
        output_dir="./roberta-sentiment",
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
        report_to="wandb",
        run_name="roberta-base-run",
    )
    
    # Trainer for RoBERTa
    roberta_trainer = Trainer(
        model=roberta_model,
        args=roberta_training_args,
        train_dataset=roberta_encoded_ds["train"],
        eval_dataset=roberta_encoded_ds["validation"],
        tokenizer=roberta_tokenizer,
        data_collator=roberta_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train RoBERTa
    print("\nStarting training...")
    roberta_train_result = roberta_trainer.train()
    print("\nRoBERTa training finished.")
    
    # Evaluation
    print("\n=== RoBERTa Validation set results ===")
    roberta_val_results = roberta_trainer.evaluate(roberta_encoded_ds["validation"])
    for k, v in roberta_val_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    print("\n=== RoBERTa Test set results ===")
    roberta_test_results = roberta_trainer.evaluate(roberta_encoded_ds["test"])
    for k, v in roberta_test_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    # Detailed evaluation and visualization
    print("\n=== Generating visualizations ===")
    evaluate_and_visualize(
        roberta_trainer,
        roberta_encoded_ds["test"],
        "RoBERTa-base",
        save_dir="./all_png/roberta_base"
    )


if __name__ == "__main__":
    main()


