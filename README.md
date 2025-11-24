# Efficient Sentiment Classification for Twitter (ECE 684 Final Project)

# Author: Bohao (Lexi) Yang (by95); Bingkun Wang (bw276); Yuzhao (Clair) Tan (yt221)

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
