"""
Train BERT-base model for sentiment analysis.
"""

import os
import numpy as np
import random
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
    LABEL_MAP,
    ID2LABEL,
    LABEL2ID,
    evaluate_and_visualize,
)


def main():
    """Train BERT-base model."""
    # Set seeds
    set_seeds(42)
    
    # Set wandb environment
    os.environ["WANDB_PROJECT"] = "tweet-eval-sentiment"
    os.environ["WANDB_WATCH"] = "false"
    
    # Load dataset
    print("Loading dataset...")
    ds = load_tweet_eval_dataset()
    
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]
    
    print(f"Train: {train_ds}")
    print(f"Validation: {val_ds}")
    print(f"Test: {test_ds}")
    
    # Tokenizer and preprocessing
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=128,
        )
    
    encoded_ds = ds.map(preprocess_function, batched=True)
    encoded_ds = encoded_ds.remove_columns(["text"])
    encoded_ds = encoded_ds.rename_column("label", "labels")
    encoded_ds.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Load BERT model
    print(f"\nLoading {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bert-sentiment",
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
    )
    
    # Build Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    print("\nTraining finished.")
    
    # Evaluation
    print("\n=== Validation set results (BERT baseline) ===")
    val_results = trainer.evaluate(encoded_ds["validation"])
    for k, v in val_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    print("\n=== Test set results (BERT baseline) ===")
    test_results = trainer.evaluate(encoded_ds["test"])
    for k, v in test_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    # Detailed evaluation and visualization
    print("\n=== Generating visualizations ===")
    evaluate_and_visualize(
        trainer,
        encoded_ds["test"],
        "BERT-base",
        save_dir="./all_png/bert_base"
    )
    
    # Show some random qualitative examples
    print("\n=== Random Qualitative Examples ===")
    raw_test = ds["test"]
    pred_output = trainer.predict(encoded_ds["test"])
    test_preds = np.argmax(pred_output.predictions, axis=-1)
    
    indices = random.sample(range(len(raw_test)), 5)
    for idx in indices:
        text = raw_test[idx]["text"]
        true_label = LABEL_MAP[raw_test[idx]["label"]]
        pred_label = LABEL_MAP[int(test_preds[idx])]
        print(f"\nTweet: {text}")
        print(f"True label: {true_label}")
        print(f"Pred label: {pred_label}")


if __name__ == "__main__":
    main()


