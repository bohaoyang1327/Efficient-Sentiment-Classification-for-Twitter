"""
Exploratory Data Analysis (EDA) for TweetEval Sentiment Dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import re
from datasets import load_dataset
from utils import set_seeds, LABEL_MAP


def plot_label_distribution(df, title, save_path=None):
    """Plot label distribution."""
    counts = df["label"].value_counts().sort_index()
    labels = [LABEL_MAP[i] for i in counts.index]
    
    plt.figure()
    plt.bar(labels, counts.values)
    plt.title(title)
    plt.xlabel("label")
    plt.ylabel("count")
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def basic_clean(text: str) -> str:
    """Basic text cleaning - keep @user and emoji."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def show_random_examples(df, n_per_label=3, random_state=42):
    """Show random examples for each label."""
    random.seed(random_state)
    for label_id, label_name in LABEL_MAP.items():
        subset = df[df["label"] == label_id]
        print(f"\n===== Label {label_id} ({label_name}) examples =====")
        samples = subset.sample(
            n=min(n_per_label, len(subset)),
            random_state=random_state
        )
        for text in samples["text"].tolist():
            print("-", text)


def main():
    """Run EDA analysis."""
    # Set seeds
    set_seeds(42)
    
    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]
    
    train_df = pd.DataFrame(train_ds)
    val_df = pd.DataFrame(val_ds)
    test_df = pd.DataFrame(test_ds)
    
    # EDA: Label distribution
    print("\n=== Label Distribution ===")
    plot_label_distribution(train_df, "Label distribution in TRAIN split", 
                          save_path="./figs/traindist.png")
    plot_label_distribution(val_df, "Label distribution in VALIDATION split",
                          save_path="./figs/validdist.png")
    plot_label_distribution(test_df, "Label distribution in TEST split",
                          save_path="./figs/testdist.png")
    
    # EDA: Token length
    print("\n=== Token Length Analysis ===")
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        df[f"{name}_token_len"] = df["text"].apply(lambda x: len(str(x).split()))
    
    plt.figure()
    plt.hist(train_df["train_token_len"], bins=50)
    plt.title("Token length distribution in TRAIN split")
    plt.xlabel("number of tokens (whitespace split)")
    plt.ylabel("count")
    plt.savefig("./figs/lengthdist.png", dpi=300)
    plt.close()
    
    print("\nTrain token length stats:")
    print(train_df["train_token_len"].describe())
    
    # EDA: Character length
    print("\n=== Character Length Analysis ===")
    train_df["char_len"] = train_df["text"].apply(lambda x: len(str(x)))
    
    plt.figure()
    plt.hist(train_df["char_len"], bins=50)
    plt.title("Character length distribution in TRAIN split")
    plt.xlabel("number of characters")
    plt.ylabel("count")
    plt.savefig("./figs/chardist.png", dpi=300)
    plt.close()
    
    print("\nCharacter length statistics:")
    print(train_df["char_len"].describe())
    
    # Data cleaning
    print("\n=== Data Cleaning ===")
    train_df["clean_text"] = train_df["text"].apply(basic_clean)
    
    print("\n===== Before/After cleaning (first 3 samples) =====")
    for i in range(3):
        print(f"\n--- Sample {i} ---")
        print("Original:", train_df.loc[i, "text"])
        print("Cleaned :", train_df.loc[i, "clean_text"])
    
    # Show random examples
    print("\n=== Random Examples from TRAIN split ===")
    show_random_examples(train_df, n_per_label=4, random_state=42)
    
    print("\nEDA completed!")


if __name__ == "__main__":
    main()


