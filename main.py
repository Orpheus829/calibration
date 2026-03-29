"""
Main Pipeline
Runs all stages in order:
  1. Load CNN/DailyMail
  2. Fine-tune BART-base
  3. Generate summaries (uncalibrated + uncertainty-aware)
  4. Detect hallucinations (BERTScore + NLI)
  5. Fit temperature scaling on validation set
  6. Run full ablation + generate plots

Usage:
    python main.py              # full pipeline
    python main.py --skip-train # skip fine-tuning, load from checkpoint
"""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm

from data.dataset import get_tokenized_splits, load_splits, MODEL_NAME
from models.summarizer import fine_tune, generate_with_scores
from calibration.methods import TemperatureScaler, uncertainty_aware_generate
from calibration.metrics import compute_all_metrics
from evaluation.hallucination import (
    NLIConsistencyChecker,
    flatten_confidences_and_labels,
)
from experiments.ablations.run_ablation import run_full_ablation

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
N_TEST         = 200   # number of test articles to evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip fine-tuning and load from checkpoint")
    parser.add_argument("--n-test", type=int, default=N_TEST,
                        help="Number of test samples to evaluate")
    return parser.parse_args()


def load_model_from_checkpoint(tokenizer=None):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    ckpt_path = f"{CHECKPOINT_DIR}/bart-finetuned/final"
    print(f"Loading model from {ckpt_path}...")
    model     = AutoModelForSeq2SeqLM.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    return model, tokenizer


def main():
    args = parse_args()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Device: {DEVICE}")

    # ── Stage 1: Data ─────────────────────────────────────────────────────────
    print("\n── Stage 1: Loading Data ──")
    tokenizer, train_tok, val_tok, test_tok, test_raw = get_tokenized_splits(
        train_size=5000, val_size=500, test_size=500
    )

    # ── Stage 2: Fine-tuning ──────────────────────────────────────────────────
    print("\n── Stage 2: Fine-tuning ──")
    if args.skip_train and os.path.exists(f"{CHECKPOINT_DIR}/bart-finetuned/final"):
        model, tokenizer = load_model_from_checkpoint()
    else:
        model = fine_tune(train_tok, val_tok, tokenizer)

    # ── Stage 3A: Generate uncalibrated summaries ─────────────────────────────
    print(f"\n── Stage 3A: Generating uncalibrated summaries (n={args.n_test}) ──")
    test_articles  = test_raw["article"][:args.n_test]
    test_highlights = test_raw["highlights"][:args.n_test]

    uncalibrated_summaries = generate_with_scores(
        model, tokenizer, test_articles,
        device=DEVICE, num_beams=4,
    )

    # ── Stage 3B: Generate uncertainty-aware summaries ────────────────────────
    print("\n── Stage 3B: Generating uncertainty-aware summaries ──")
    uncertainty_summaries = []
    for article in tqdm(test_articles, desc="Uncertainty-aware decoding"):
        result = uncertainty_aware_generate(
            model, tokenizer, article,
            device=DEVICE, num_beams=4, entropy_penalty=0.3,
        )
        uncertainty_summaries.append(result)

    # ── Stage 4: Hallucination Detection ─────────────────────────────────────
    print("\n── Stage 4: Hallucination Detection ──")
    checker = NLIConsistencyChecker(device=DEVICE)

    uncalibrated_labeled = checker.label_summaries(uncalibrated_summaries, test_articles)
    uncertainty_labeled  = checker.label_summaries(uncertainty_summaries,  test_articles)

    # ── Stage 5: Fit Temperature Scaling on Validation Set ───────────────────
    print("\n── Stage 5: Temperature Scaling ──")

    # Generate val summaries for fitting temperature
    val_articles = load_splits(val_size=200)[1]["article"][:200]  # type: ignore
    val_summaries = generate_with_scores(
        model, tokenizer, val_articles, device=DEVICE
    )
    val_labeled = checker.label_summaries(val_summaries, val_articles)
    val_confs, val_labels = flatten_confidences_and_labels(val_labeled)

    scaler = TemperatureScaler()
    scaler.fit(val_confs, val_labels)

    # Apply temperature scaling to test uncalibrated summaries
    temperature_summaries = []
    for summary_dict in uncalibrated_labeled:
        scaled_confs = scaler.scale_confidences(summary_dict["sentence_confidences"])
        temperature_summaries.append({
            **summary_dict,
            "sentence_confidences": scaled_confs,
            "method": "temperature_scaling",
        })

    # ── Stage 6: Full Ablation ────────────────────────────────────────────────
    print("\n── Stage 6: Ablation ──")
    results = run_full_ablation(
        uncalibrated_summaries=uncalibrated_labeled,
        temperature_summaries=temperature_summaries,
        uncertainty_summaries=uncertainty_labeled,
    )

    # Save all summaries for inspection
    out_path = os.path.join(CHECKPOINT_DIR, "all_summaries.json")
    with open(out_path, "w") as f:
        json.dump({
            "uncalibrated": uncalibrated_labeled[:20],    # first 20 for inspection
            "temperature":  temperature_summaries[:20],
            "uncertainty":  uncertainty_labeled[:20],
        }, f, indent=2, default=str)
    print(f"\nSample summaries saved → {out_path}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
