# Calibration-Aware Abstractive Summarization

> **Research question:** Does sentence-level miscalibration predict hallucination risk in abstractive summarization? Does uncertainty-aware decoding produce better-calibrated and more factually grounded summaries than post-hoc temperature scaling?

---

## Motivation

BART and similar models are systematically overconfident — they assign high confidence to sentences even when they hallucinate. This project investigates whether **calibration quality predicts factual faithfulness**, and compares two fundamentally different approaches to fixing miscalibration:

- **Post-hoc (Method B):** Temperature scaling applied after generation
- **Decoding-time (Method C):** Entropy-penalized beam search during generation

---

## Pipeline

```
[CNN/DailyMail]
      ↓
[Fine-tune BART-base]
      ↓
[Generate summaries — 3 conditions]
  A. Uncalibrated (standard beam search)
  B. Temperature Scaling (post-hoc)
  C. Uncertainty-Aware Decoding (entropy penalty at generation time)
      ↓
[Hallucination Detection]
  - BERTScore F1 per sentence vs source
  - NLI entailment check (DeBERTa)
      ↓
[Calibration Evaluation]
  - ECE, MCE, Brier Score per condition
  - Reliability diagrams
      ↓
[Core Analysis]
  - Spearman correlation: confidence ↔ factual consistency
  - Wilcoxon test: B vs C calibration quality
  - ECE vs hallucination rate scatter plot
```

---

## Key Results (fill after running)

| Method | ECE ↓ | MCE ↓ | Brier ↓ | Hallucination % ↓ | Spearman r ↑ |
|---|---|---|---|---|---|
| Uncalibrated | — | — | — | — | — |
| Temperature Scaling | — | — | — | — | — |
| Uncertainty-Aware | — | — | — | — | — |

---

## Setup

```bash
pip install torch transformers datasets bert-score scipy matplotlib tqdm accelerate
```

## Usage

```bash
# Full pipeline
python main.py

# Skip fine-tuning (load from checkpoint)
python main.py --skip-train

# Evaluate on fewer samples (faster)
python main.py --skip-train --n-test 50
```

## Execution order (if running stages separately)

```bash
python -m data.dataset                          # verify data loads
python main.py                                  # full pipeline
python -m experiments.ablations.run_ablation    # regenerate plots
```

---

## Project Structure

```
calibrated-summarization/
├── data/
│   └── dataset.py              # CNN/DailyMail loading + tokenization
├── models/
│   └── summarizer.py           # BART fine-tuning + confidence extraction
├── calibration/
│   ├── methods.py              # Temperature scaling + uncertainty-aware decoding
│   └── metrics.py              # ECE, MCE, Brier Score, reliability diagrams
├── evaluation/
│   └── hallucination.py        # BERTScore + NLI consistency labeling
├── experiments/
│   └── ablations/
│       └── run_ablation.py     # Full comparison + significance tests + plots
├── main.py                     # End-to-end pipeline runner
├── paper_notes.md              # Research connections
└── requirements.txt
```

---

## Research Connections

- **Wang & Stengel-Eskin (ICLR 2026)** — calibrating generation, complementary approach
- **Stengel-Eskin et al. (NeurIPS 2024)** — calibration via pragmatic reasoning
- **Guo et al. (ICML 2017)** — temperature scaling foundation
- **Maynez et al. (ACL 2020)** — hallucination rates in BART summarization

See `paper_notes.md` for full positioning.
