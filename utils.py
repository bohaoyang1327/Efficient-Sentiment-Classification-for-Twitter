"""
Shared utilities for NLP sentiment analysis project.
Contains common functions for data loading, metrics, and visualization.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
import torch
import random


# Set random seeds for reproducibility
def set_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Label mappings
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
ID2LABEL = {i: LABEL_MAP[i] for i in LABEL_MAP}
LABEL2ID = {v: k for k, v in LABEL_MAP.items()}


def load_tweet_eval_dataset():
    """Load the TweetEval sentiment dataset."""
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    return ds


def compute_metrics(eval_pred):
    """Compute accuracy and macro-F1 metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    
    return {"accuracy": acc, "macro_f1": macro_f1}


def plot_training_curves(trainer, model_name, save_dir="./figs"):
    """Plot training and validation loss curves, and macro-F1 curve."""
    logs = trainer.state.log_history
    df_logs = pd.DataFrame(logs)
    
    # Separate training and evaluation logs
    df_train = df_logs[df_logs["loss"].notnull()]
    df_eval = df_logs[df_logs["eval_loss"].notnull()]
    
    # Loss curve
    plt.figure()
    plt.plot(df_train["epoch"], df_train["loss"], label="train loss")
    plt.plot(df_eval["epoch"], df_eval["eval_loss"], label="eval loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({model_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_loss.png", dpi=300)
    plt.close()
    
    # Macro-F1 curve
    if "eval_macro_f1" in df_eval.columns:
        plt.figure()
        plt.plot(df_eval["epoch"], df_eval["eval_macro_f1"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Macro F1")
        plt.title(f"Validation Macro-F1 over Epochs ({model_name})")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_macro_f1.png", dpi=300)
        plt.close()


def plot_confusion_matrix(test_labels, test_preds, model_name, save_dir="./figs"):
    """Plot confusion matrix."""
    cm = confusion_matrix(test_labels, test_preds)
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    )
    
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name} (TweetEval Sentiment)")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300)
    plt.close()


def plot_per_class_metrics(test_labels, test_preds, model_name, save_dir="./figs"):
    """Plot per-class precision, recall, and F1-score."""
    report_dict = classification_report(
        test_labels,
        test_preds,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
        output_dict=True,
    )
    
    cls_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    metrics = ["precision", "recall", "f1-score"]
    
    data = {m: [report_dict[cls][m] for cls in cls_names] for m in metrics}
    df_metrics = pd.DataFrame(data, index=cls_names)
    
    for m in metrics:
        plt.figure()
        df_metrics[m].plot(kind="bar")
        plt.ylim(0, 1.0)
        plt.title(f"{m.capitalize()} per class ({model_name})")
        plt.xlabel("Class")
        plt.ylabel(m.capitalize())
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model_name.lower().replace(' ', '_')}_{m}_per_class.png", dpi=300)
        plt.close()


def evaluate_and_visualize(trainer, test_dataset, model_name, save_dir="./figs"):
    """Evaluate model and create visualizations."""
    # Get predictions
    pred_output = trainer.predict(test_dataset)
    test_logits = pred_output.predictions
    test_labels = pred_output.label_ids
    test_preds = np.argmax(test_logits, axis=-1)
    
    # Print classification report
    print(f"\n=== Classification report on TEST set ({model_name}) ===")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
        )
    )
    
    # Create visualizations
    plot_confusion_matrix(test_labels, test_preds, model_name, save_dir)
    plot_per_class_metrics(test_labels, test_preds, model_name, save_dir)
    plot_training_curves(trainer, model_name, save_dir)
    
    return test_labels, test_preds


