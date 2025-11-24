"""
Train BERTweet with LoRA for sentiment analysis.
"""

import os
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
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
    """Train BERTweet with LoRA."""
    # Set seeds
    set_seeds(42)
    
    # Set wandb environment
    os.environ["WANDB_DISABLED"] = "true"
    
    # Load dataset
    print("Loading dataset...")
    ds = load_tweet_eval_dataset()
    
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
    bertweet_model = AutoModelForSequenceClassification.from_pretrained(
        bertweet_model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # LoRA configuration & wrapping
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query", "value"],
    )
    
    bertweet_model = get_peft_model(bertweet_model, lora_config)
    bertweet_model.print_trainable_parameters()
    
    # Training arguments
    bertweet_training_args = TrainingArguments(
        output_dir="./bertweet-sentiment-lora",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=1e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
    )
    
    # Trainer
    bertweet_trainer = Trainer(
        model=bertweet_model,
        args=bertweet_training_args,
        train_dataset=bertweet_encoded_ds["train"],
        eval_dataset=bertweet_encoded_ds["validation"],
        tokenizer=bertweet_tokenizer,
        data_collator=bertweet_data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    bertweet_train_result = bertweet_trainer.train()
    print("\nBERTweet + LoRA training finished.")
    
    # Evaluation
    print("\n=== BERTweet + LoRA Validation set results ===")
    bertweet_val_results = bertweet_trainer.evaluate(bertweet_encoded_ds["validation"])
    for k, v in bertweet_val_results.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {v:.4f}")
    
    print("\n=== BERTweet + LoRA Test set results ===")
    bertweet_test_results = bertweet_trainer.evaluate(bertweet_encoded_ds["test"])
    for k, v in bertweet_test_results.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {v:.4f}")
    
    # Detailed evaluation and visualization
    print("\n=== Generating visualizations ===")
    evaluate_and_visualize(
        bertweet_trainer,
        bertweet_encoded_ds["test"],
        "BERTweet + LoRA",
        save_dir="./all_png/bertweet_lora"
    )


if __name__ == "__main__":
    main()


