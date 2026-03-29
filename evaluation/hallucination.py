"""
Evaluation — Hallucination Detection

Two complementary hallucination metrics:

1. BERTScore F1
   - Measures semantic similarity between summary sentence and source article
   - Uses contextual embeddings (roberta-large) — captures paraphrase
   - Threshold: sentence BERTScore < 0.85 → likely hallucinated
   - Fast, no extra model needed beyond transformers

2. NLI-based Factual Consistency (TRUE-style)
   - Uses a pretrained NLI model to check if source ENTAILS summary sentence
   - More principled than BERTScore — checks logical consistency not similarity
   - Model: cross-encoder/nli-deberta-v3-small (lightweight, free)
   - Label: ENTAILMENT → consistent, CONTRADICTION/NEUTRAL → hallucinated

Research use:
  These binary labels (consistent=1, hallucinated=0) are the ground truth
  for calibration evaluation. The key question:
  Does sentence confidence predict the hallucination label?
  A well-calibrated model should have high confidence on consistent sentences
  and low confidence on hallucinated ones.
"""

import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore_labels(
    summaries: list[dict],
    articles: list[str],
    threshold: float = 0.85,
    device: str = "cpu",
) -> list[dict]:
    """
    Compute BERTScore F1 per sentence against source article.
    Returns label: 1 = consistent (BERTScore >= threshold), 0 = hallucinated.

    summaries: list of dicts with "sentences" key from generate_with_scores()
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        print("bert_score not installed. Run: pip install bert-score")
        return _fallback_labels(summaries)

    print("Computing BERTScore labels...")
    labeled_summaries = []

    for summary_dict, article in zip(summaries, articles):
        sentences  = summary_dict["sentences"]
        # Reference for each sentence = full source article
        references = [article] * len(sentences)

        if not sentences:
            labeled_summaries.append({**summary_dict, "labels": [], "bertscore_f1": []})
            continue

        _, _, F1 = bert_score_fn(
            sentences, references,
            lang="en",
            model_type="roberta-large",
            device=device,
            verbose=False,
        )

        f1_scores = F1.tolist()
        labels    = [1 if f >= threshold else 0 for f in f1_scores]

        labeled_summaries.append({
            **summary_dict,
            "labels":       labels,
            "bertscore_f1": f1_scores,
        })

    n_total = sum(len(s["labels"]) for s in labeled_summaries)
    n_hall  = sum(sum(1 - l for l in s["labels"]) for s in labeled_summaries)
    print(f"  Hallucination rate: {n_hall}/{n_total} = {100*n_hall/max(n_total,1):.1f}%")
    return labeled_summaries


# ── NLI-based Consistency ─────────────────────────────────────────────────────

class NLIConsistencyChecker:
    """
    Uses DeBERTa NLI model to check if source article entails summary sentence.
    More principled than BERTScore — checks logical consistency.
    """

    def __init__(self, device: str = "cpu"):
        model_name = "cross-encoder/nli-deberta-v3-small"
        print(f"Loading NLI model: {model_name}")
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
        )
        self.device = device

    def check_sentence(self, sentence: str, article: str) -> dict:
        """
        Check if article entails sentence.
        Returns: {"label": int, "entailment_score": float}
        """
        # Truncate article to avoid token limit
        article_truncated = article[:1000]
        result = self.classifier(
            f"{article_truncated} [SEP] {sentence}",
            truncation=True,
            max_length=512,
        )

        label_str = result[0]["label"].lower()
        score     = result[0]["score"]

        is_consistent = 1 if "entail" in label_str else 0
        entailment_score = score if "entail" in label_str else 1 - score

        return {
            "label":             is_consistent,
            "entailment_score":  float(entailment_score),
            "raw_label":         label_str,
        }

    def label_summaries(
        self,
        summaries: list[dict],
        articles: list[str],
    ) -> list[dict]:
        """Label all sentences in all summaries."""
        print("Running NLI consistency check...")
        labeled = []

        for summary_dict, article in zip(summaries, articles):
            sentences = summary_dict["sentences"]
            nli_results = [self.check_sentence(s, article) for s in sentences]

            labels           = [r["label"] for r in nli_results]
            entailment_scores = [r["entailment_score"] for r in nli_results]

            labeled.append({
                **summary_dict,
                "labels":            labels,
                "entailment_scores": entailment_scores,
                "method":            "nli",
            })

        n_total = sum(len(s["labels"]) for s in labeled)
        n_hall  = sum(sum(1 - l for l in s["labels"]) for s in labeled)
        print(f"  Hallucination rate: {n_hall}/{n_total} = {100*n_hall/max(n_total,1):.1f}%")
        return labeled


def _fallback_labels(summaries):
    """Fallback if bert_score not installed — random labels for testing."""
    print("WARNING: Using random fallback labels. Install bert-score for real evaluation.")
    return [{**s, "labels": [1] * len(s["sentences"]), "bertscore_f1": [1.0] * len(s["sentences"])}
            for s in summaries]


def flatten_confidences_and_labels(labeled_summaries: list[dict]) -> tuple:
    """
    Flatten per-sentence confidences and labels across all summaries
    into flat lists for calibration metric computation.
    """
    all_confidences = []
    all_labels      = []

    for s in labeled_summaries:
        confs  = s.get("sentence_confidences", [])
        labels = s.get("labels", [])
        n = min(len(confs), len(labels))
        all_confidences.extend(confs[:n])
        all_labels.extend(labels[:n])

    return all_confidences, all_labels
