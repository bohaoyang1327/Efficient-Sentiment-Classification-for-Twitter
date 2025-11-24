"""
Train BERT-base with LoRA for sentiment analysis.
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
    """Train BERT-base with LoRA."""
    # Set seeds
    set_seeds(42)
    
    # Set wandb environment
    os.environ["WANDB_PROJECT"] = "tweet-eval-sentiment"
    os.environ["WANDB_WATCH"] = "false"
    
    # Load dataset
    print("Loading dataset...")
    ds = load_tweet_eval_dataset()
    
    # Tokenizer and preprocessing (reuse BERT tokenizer)
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
    
    # Load base BERT model
    print(f"\nLoading {model_name}...")
    lora_base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],  # attention projection layers
    )
    
    # Wrap base model with LoRA
    lora_model = get_peft_model(lora_base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    # Training arguments
    lora_training_args = TrainingArguments(
        output_dir="./lora-bert-sentiment",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="wandb",
        run_name="bert-lora-run",
    )
    
    # Trainer for LoRA-BERT
    lora_trainer = Trainer(
        model=lora_model,
        args=lora_training_args,
        train_dataset=encoded_ds["train"],
        eval_dataset=encoded_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train LoRA-BERT
    print("\nStarting training...")
    lora_train_result = lora_trainer.train()
    print("\nLoRA-BERT training finished.")
    
    # Evaluation
    print("\n=== LoRA-BERT Validation set results ===")
    lora_val_results = lora_trainer.evaluate(encoded_ds["validation"])
    for k, v in lora_val_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    print("\n=== LoRA-BERT Test set results ===")
    lora_test_results = lora_trainer.evaluate(encoded_ds["test"])
    for k, v in lora_test_results.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
    
    # Detailed evaluation and visualization
    print("\n=== Generating visualizations ===")
    evaluate_and_visualize(
        lora_trainer,
        encoded_ds["test"],
        "LoRA-BERT",
        save_dir="./all_png/bert_lora"
    )


if __name__ == "__main__":
    main()


