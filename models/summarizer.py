"""
Model — BART Fine-tuning
Fine-tunes BART-base on CNN/DailyMail summarization.
Modified to output token-level log probabilities needed for
sentence-level confidence estimation in calibration stage.

Key design decision: we don't modify the architecture —
we extract confidence from the model's own output scores,
which is more principled than adding a separate classification head.
"""

import os
import random
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from data.dataset import MODEL_NAME, MAX_TARGET_LEN

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_training_args(output_dir: str = "checkpoints/bart-finetuned") -> Seq2SeqTrainingArguments:
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,       # fits on free Colab T4
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,       # effective batch size = 16
        warmup_steps=200,
        weight_decay=0.01,
        # logging_dir removed in v5.2
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LEN,
        fp16=torch.cuda.is_available(),      # half precision on GPU
        seed=SEED,
        report_to="none",                    # disable wandb
    )


def fine_tune(train_tok, val_tok, tokenizer):
    """Fine-tune BART-base on tokenized CNN/DailyMail splits."""
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    args = get_training_args()
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("Fine-tuning BART-base...")
    trainer.train()

    model.save_pretrained(f"{CHECKPOINT_DIR}/bart-finetuned/final")
    tokenizer.save_pretrained(f"{CHECKPOINT_DIR}/bart-finetuned/final")
    print(f"Model saved → {CHECKPOINT_DIR}/bart-finetuned/final")
    return model


def generate_with_scores(
    model,
    tokenizer,
    articles: list[str],
    device: str = "cpu",
    num_beams: int = 4,
) -> list[dict]:
    """
    Generate summaries and extract token-level log probabilities.
    This is the foundation for sentence-level confidence estimation.

    Returns list of dicts:
    {
        "summary": str,
        "sentences": list[str],
        "token_log_probs": list[float],   # per token
        "sentence_confidences": list[float]  # mean exp(log_prob) per sentence
    }
    """
    model.eval()
    model.to(device)
    results = []

    for article in articles:
        inputs = tokenizer(
            article,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=128,
                output_scores=True,         # get token scores
                return_dict_in_generate=True,
            )

        # Decode full summary
        summary = tokenizer.decode(
            output.sequences[0],
            skip_special_tokens=True,
        )

        # Extract token-level log probabilities
        # output.scores: tuple of [vocab_size] tensors, one per generated token
        token_log_probs = []
        if output.scores:
            n_input = inputs["input_ids"].shape[1]
            generated_ids = output.sequences[0][n_input:]
            for step_idx, step_scores in enumerate(output.scores):
                if step_idx >= len(generated_ids):
                    break
                log_probs = torch.log_softmax(step_scores[0], dim=-1)
                chosen_token = generated_ids[step_idx].item()
                if 0 <= chosen_token < log_probs.shape[-1]:
                    token_log_probs.append(log_probs[chosen_token].item())

        # Split into sentences and assign confidence per sentence
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        sentence_confidences = _assign_sentence_confidences(
            sentences, token_log_probs, tokenizer
        )

        results.append({
            "summary": summary,
            "sentences": sentences,
            "token_log_probs": token_log_probs,
            "sentence_confidences": sentence_confidences,
        })

    return results


def _assign_sentence_confidences(
    sentences: list[str],
    token_log_probs: list[float],
    tokenizer,
) -> list[float]:
    """
    Assign a confidence score to each sentence by averaging the
    exponentiated log probs of tokens in that sentence.
    Confidence = mean P(token) across all tokens in sentence.
    Range: [0, 1]. Higher = more confident.
    """
    if not token_log_probs or not sentences:
        return [0.5] * len(sentences)

    confidences = []
    token_idx = 0

    for sentence in sentences:
        n_tokens = len(tokenizer.encode(sentence, add_special_tokens=False))
        sent_log_probs = token_log_probs[token_idx: token_idx + n_tokens]
        token_idx += n_tokens

        if sent_log_probs:
            # Convert log probs to probs, take mean
            conf = float(np.mean(np.exp(sent_log_probs)))
        else:
            conf = 0.5
        confidences.append(min(max(conf, 0.0), 1.0))  # clamp to [0,1]

    return confidences
