
````markdown
# Efficient Sentiment Classification for Twitter (ECE 684 Final Project)

# Author: Bohao (Lexi) Yang (by95); Bingkun Wang (bw276); Yuzhao (Clair) Tan (yt221)

This repository contains our full workflow for the ECE 684 final project.  
All model development and experiments were first trained and tested in a **Python Jupyter notebook**, and then organized into modular Python scripts for easier reuse and clearer structure.

**Step 1 — NLP problem:**  
We study three-class sentiment classification on tweets (negative / neutral / positive). We use the TweetEval “sentiment” benchmark, which is designed for noisy Twitter text.

**Step 2 — Language-model-based solution + novelty:**  
We fine-tune and compare several Transformer models for the same task:  
1. BERT-base (general-domain baseline)  
2. RoBERTa-base (stronger general-domain baseline)  
3. BERTweet-base (domain-specific model pretrained on tweets)  
To improve efficiency, we also apply LoRA parameter-efficient fine-tuning to all three backbones and compare full fine-tuning vs. LoRA under the same training budget.  
Finally, to address mild class imbalance, we test class-weighted loss + label smoothing on BERTweet.

**Step 3a — Synthetic data (not required here):**  
Our models are discriminative classifiers, so Step 3a does not apply.

**Step 3b — Real data + evaluation:**  
All models are trained and evaluated on the real TweetEval dataset with accuracy, macro-F1, and per-class analysis.

**Step 4 — Pros/cons:**  
We discuss model quality vs. computation cost, interpretability limits, and class-imbalance behavior in the report.


## Data (how to get it)

We use the official HuggingFace split of TweetEval sentiment:  
`cardiffnlp/tweet_eval`, configuration `sentiment`.

You do **not** need to manually download files. The notebook loads the dataset automatically:

```python
from datasets import load_dataset
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
train_ds, val_ds, test_ds = ds["train"], ds["validation"], ds["test"]
````

The following sections describe the modular script structure that mirrors the workflow we originally carried out in Jupyter.

# NLP Sentiment Analysis — Modular Scripts

This directory contains modular Python scripts converted from the original Jupyter notebook (`NLP_model.ipynb`).
These scripts follow the same logic as the notebook but separate each part for clarity.

## Structure

### Core Modules

* **`utils.py`**: Shared utilities module containing:

  * Common functions (seed setting, data loading)
  * Metrics computation (`compute_metrics`)
  * Visualization functions (confusion matrix, training curves, per-class metrics)
  * Label mappings and constants

### Scripts

1. **`eda.py`**: Exploratory Data Analysis

   * Label distribution analysis
   * Token and character length statistics
   * Data cleaning examples
   * Random sample visualization

2. **`train_bert_base.py`**: Train BERT-base model

   * Baseline BERT model training
   * Full fine-tuning approach

3. **`train_bert_lora.py`**: Train BERT-base with LoRA

   * Parameter-efficient fine-tuning using LoRA
   * Reduced trainable parameters

4. **`train_roberta_base.py`**: Train RoBERTa-base model

   * RoBERTa baseline training

5. **`train_roberta_lora.py`**: Train RoBERTa-base with LoRA

   * RoBERTa with LoRA fine-tuning

6. **`train_bertweet.py`**: Train BERTweet model

   * Twitter-specific pre-trained model
   * Uses normalization for Twitter text

7. **`train_bertweet_lora.py`**: Train BERTweet with LoRA

   * BERTweet with LoRA fine-tuning

8. **`train_bertweet_weighted.py`**: Train BERTweet with class weights and label smoothing

   * Handles class imbalance with weighted loss
   * Includes label smoothing for regularization

## Usage

### Run EDA

```bash
python eda.py
```

### Train Models

```bash
# BERT models
python train_bert_base.py
python train_bert_lora.py

# RoBERTa models
python train_roberta_base.py
python train_roberta_lora.py

# BERTweet models
python train_bertweet.py
python train_bertweet_lora.py
python train_bertweet_weighted.py
```

## Output

Each training script will:

1. Train the model and save checkpoints to `./{model-name}-sentiment/`
2. Evaluate on validation and test sets
3. Generate visualizations in `./all_png/{model_name}/`, including:

   * Confusion matrix
   * Per-class precision, recall, F1-score
   * Training/validation loss curves
   * Macro-F1 over epochs

## Dependencies

Required packages:

* `datasets`
* `transformers`
* `torch`
* `peft` (for LoRA models)
* `scikit-learn`
* `matplotlib`
* `pandas`
* `numpy`

Install with:

```bash
pip install datasets transformers torch peft scikit-learn matplotlib pandas numpy
```

## Notes

* All scripts use random seed 42 for reproducibility
* Models are trained for 3 epochs by default
* Batch sizes: 32 for training, 64 for evaluation
* Max sequence length: 128 tokens
* WandB logging is configured (can be disabled by setting `WANDB_DISABLED=true`)
* Original model training and testing were first completed in a **Python Jupyter notebook**, and the scripts here follow the same experimental setup.

```

If you'd like, I can also help format this to match your GitHub styling preferences (indentation, badges, table of contents, etc.).
```
