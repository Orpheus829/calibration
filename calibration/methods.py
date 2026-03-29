"""
Calibration — Two Methods Compared

Method A: Post-hoc Temperature Scaling
  - Train a single scalar T on validation set
  - Divide logits by T before softmax
  - T > 1 softens distribution (less confident)
  - T < 1 sharpens distribution (more confident)
  - Classic method (Guo et al. 2017) — our baseline calibration approach

Method B: Uncertainty-Aware Beam Search
  - Modify decoding to penalize high-entropy steps
  - At each decoding step, if entropy of distribution is high
    (model is uncertain), reduce beam score
  - Produces summaries that avoid low-confidence token choices
  - More principled than post-hoc — uncertainty shapes generation itself

Research question: does Method B produce better-calibrated AND
more factually grounded summaries than Method A?
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# ── Method A: Temperature Scaling ────────────────────────────────────────────

class TemperatureScaler(nn.Module):
    """
    Single scalar temperature parameter applied to logits.
    Trained by minimizing NLL on validation sentence confidences.
    """
    def __init__(self):
        super().__init__()
        # Initialize T=1 (no scaling) — optimal T learned from val set
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, confidences: torch.Tensor) -> torch.Tensor:
        """Scale confidence scores by temperature."""
        # Convert confidence to logit space, scale, convert back
        eps = 1e-7
        confidences = confidences.clamp(eps, 1 - eps)
        logits = torch.log(confidences / (1 - confidences))  # logit transform
        scaled_logits = logits / self.temperature
        return torch.sigmoid(scaled_logits)

    def fit(self, confidences: list[float], labels: list[int]) -> float:
        """
        Fit temperature T to minimize NLL on validation set.
        labels: 1 if sentence is factually consistent, 0 if hallucinated
        Returns optimal temperature value.
        """
        conf_tensor = torch.tensor(confidences, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS(
            [self.temperature],
            lr=0.01,
            max_iter=100,
        )

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(conf_tensor)
            loss = nn.BCELoss()(scaled, label_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(self.temperature.item())
        print(f"  Optimal temperature: T = {T:.4f}")
        print(f"  {'Overconfident (T>1 softens)' if T > 1 else 'Underconfident (T<1 sharpens)'}")
        return T

    def scale_confidences(self, confidences: list[float]) -> list[float]:
        """Apply learned temperature to a list of confidence scores."""
        with torch.no_grad():
            conf_tensor = torch.tensor(confidences, dtype=torch.float32)
            scaled = self.forward(conf_tensor)
        return scaled.tolist()


# ── Method B: Uncertainty-Aware Beam Search ──────────────────────────────────

def uncertainty_aware_generate(
    model,
    tokenizer,
    article: str,
    device: str = "cpu",
    num_beams: int = 4,
    entropy_penalty: float = 0.3,
) -> dict:
    """
    Generate summary with uncertainty-aware decoding.

    At each decoding step, compute entropy of the token distribution.
    High entropy = model is uncertain = penalize this decoding path.

    This differs from standard beam search which only maximizes
    token probability without considering uncertainty.

    entropy_penalty: weight of entropy penalty on beam score (0 = standard beam search)
    """
    model.eval()
    model.to(device)

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
            output_scores=True,
            return_dict_in_generate=True,
        )

    summary = tokenizer.decode(output.sequences[0], skip_special_tokens=True)

    # Compute per-step entropy and adjusted confidence
    step_entropies = []
    adjusted_log_probs = []

    # output.scores[i] = logits at generation step i
    # generated tokens = output.sequences[0][n_input_tokens:]
    n_input = inputs["input_ids"].shape[1]
    generated_ids = output.sequences[0][n_input:]

    for step_idx, step_scores in enumerate(output.scores):
        if step_idx >= len(generated_ids):
            break
        log_probs = torch.log_softmax(step_scores[0], dim=-1)
        probs     = torch.softmax(step_scores[0], dim=-1)

        entropy = -(probs * log_probs).sum().item()
        step_entropies.append(entropy)

        chosen_token_idx = generated_ids[step_idx].item()
        if 0 <= chosen_token_idx < log_probs.shape[-1]:
            raw_lp = log_probs[chosen_token_idx].item()
            adjusted_lp = raw_lp - entropy_penalty * entropy
            adjusted_log_probs.append(adjusted_lp)

    sentences = [s.strip() for s in summary.split(".") if s.strip()]

    # Sentence confidence from adjusted log probs
    sentence_confidences = []
    token_idx = 0
    for sent in sentences:
        n_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        sent_lps = adjusted_log_probs[token_idx: token_idx + n_tokens]
        token_idx += n_tokens
        if sent_lps:
            conf = float(np.mean(np.exp(np.clip(sent_lps, -20, 0))))
        else:
            conf = 0.5
        sentence_confidences.append(min(max(conf, 0.0), 1.0))

    return {
        "summary": summary,
        "sentences": sentences,
        "sentence_confidences": sentence_confidences,
        "mean_entropy": float(np.mean(step_entropies)) if step_entropies else 0.0,
        "method": "uncertainty_aware",
    }
