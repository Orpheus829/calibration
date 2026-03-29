"""
Data — CNN/DailyMail Loading
Loads and preprocesses CNN/DailyMail for summarization.
Uses Hugging Face datasets — no manual download needed.

Usage:
    from data.dataset import load_splits
    train, val, test = load_splits()
"""

import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# CNN/DailyMail v3.0.0 — standard benchmark split
DATASET_NAME    = "cnn_dailymail"
DATASET_VERSION = "3.0.0"

# Model backbone — BART-base balances quality vs Colab memory
MODEL_NAME = "facebook/bart-base"

# Truncation lengths — standard for CNN/DailyMail
MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128


def load_splits(
    train_size: int = 5000,   # subset for feasibility on free Colab
    val_size:   int = 500,
    test_size:  int = 500,
):
    """
    Load CNN/DailyMail splits. Subsets for compute efficiency.
    Full dataset has 287k train / 13k val / 11k test.
    5k/500/500 is sufficient for calibration research — we care about
    calibration curves, not SOTA ROUGE scores.
    """
    print("Loading CNN/DailyMail dataset...")
    dataset = load_dataset(DATASET_NAME, DATASET_VERSION)

    train = dataset["train"].shuffle(seed=SEED).select(range(train_size))
    val   = dataset["validation"].shuffle(seed=SEED).select(range(val_size))
    test  = dataset["test"].shuffle(seed=SEED).select(range(test_size))

    print(f"  Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


def tokenize_batch(batch, tokenizer):
    """
    Tokenize article + highlights for seq2seq training.
    Padding handled by DataCollator at training time.
    """
    model_inputs = tokenizer(
        batch["article"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding=False,
    )
    labels = tokenizer(
        text_target=batch["highlights"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def get_tokenized_splits(train_size=5000, val_size=500, test_size=500):
    """Full pipeline: load → tokenize → return."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train, val, test = load_splits(train_size, val_size, test_size)

    cols_to_remove = ["article", "highlights", "id"]

    train_tok = train.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )
    val_tok = val.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )
    test_tok = test.map(
        lambda b: tokenize_batch(b, tokenizer),
        batched=True,
        remove_columns=cols_to_remove,
    )

    return tokenizer, train_tok, val_tok, test_tok, test
