"""
Ablation — Full Comparison

Three conditions:
  A. Uncalibrated (raw BART beam search)
  B. Post-hoc Temperature Scaling
  C. Uncertainty-Aware Decoding

For each condition, compute:
  - ECE, MCE, Brier Score
  - Hallucination rate
  - Spearman correlation between confidence and factual consistency
  - Wilcoxon signed-rank test for statistical significance

The core research claim is tested here:
  Does uncertainty-aware decoding (C) beat post-hoc calibration (B)
  on both calibration quality AND factual grounding?
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from calibration.metrics import compute_all_metrics, plot_reliability_diagram
from evaluation.hallucination import flatten_confidences_and_labels

RESULTS_DIR = "experiments/ablations"
FIGURES_DIR = "experiments/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def run_full_ablation(
    uncalibrated_summaries: list[dict],
    temperature_summaries:  list[dict],
    uncertainty_summaries:  list[dict],
) -> dict:
    """
    Run full ablation across all three conditions.
    Each input is a list of labeled summary dicts with:
      - sentence_confidences
      - labels (1=consistent, 0=hallucinated)
    """
    conditions = {
        "Uncalibrated":          uncalibrated_summaries,
        "Temperature Scaling":   temperature_summaries,
        "Uncertainty-Aware":     uncertainty_summaries,
    }

    all_results = {}

    for name, summaries in conditions.items():
        confs, labels = flatten_confidences_and_labels(summaries)

        if not confs:
            print(f"  Skipping {name} — no data")
            continue

        metrics = compute_all_metrics(confs, labels, method_name=name)

        # Spearman correlation: confidence ↔ factual consistency
        if len(set(labels)) > 1:
            corr, pval = spearmanr(confs, labels)
        else:
            corr, pval = 0.0, 1.0

        metrics["spearman_r"]    = float(corr)
        metrics["spearman_pval"] = float(pval)
        metrics["hallucination_rate"] = float(1 - np.mean(labels))
        all_results[name] = metrics

    # Statistical significance: Temperature vs Uncertainty-Aware
    _run_significance_tests(
        uncalibrated_summaries,
        temperature_summaries,
        uncertainty_summaries,
        all_results,
    )

    # Save results
    out_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(out_path, "w") as f:
        # bin_stats contains numpy floats — convert for JSON
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                kk: (float(vv) if isinstance(vv, (np.floating, float)) else vv)
                for kk, vv in v.items()
                if kk != "bin_stats"
            }
        json.dump(serializable, f, indent=2)
    print(f"\nAblation results saved → {out_path}")

    print_results_table(all_results)
    plot_reliability_diagram(
        list(all_results.values()),
        output_path=os.path.join(FIGURES_DIR, "reliability_diagram.png"),
    )
    plot_calibration_hallucination_correlation(all_results)

    return all_results


def _run_significance_tests(uncal, temp, unc_aware, results):
    """Wilcoxon signed-rank test between conditions (paired per sentence)."""
    print("\n── Statistical Significance Tests ──")

    pairs = [
        ("Uncalibrated",        "Temperature Scaling",  uncal,  temp),
        ("Uncalibrated",        "Uncertainty-Aware",    uncal,  unc_aware),
        ("Temperature Scaling", "Uncertainty-Aware",    temp,   unc_aware),
    ]

    for name_a, name_b, sums_a, sums_b in pairs:
        confs_a, labels_a = flatten_confidences_and_labels(sums_a)
        confs_b, labels_b = flatten_confidences_and_labels(sums_b)

        n = min(len(confs_a), len(confs_b))
        if n < 5:
            continue

        # Squared errors as the paired metric
        sq_err_a = [(c - l) ** 2 for c, l in zip(confs_a[:n], labels_a[:n])]
        sq_err_b = [(c - l) ** 2 for c, l in zip(confs_b[:n], labels_b[:n])]

        try:
            stat, p = wilcoxon(sq_err_a, sq_err_b, alternative="two-sided")
            sig = "✓ significant" if p < 0.05 else "✗ not significant"
            print(f"  {name_a} vs {name_b}: W={stat:.1f}, p={p:.4f}  [{sig}]")
            results[f"sig_{name_a}_vs_{name_b}"] = {"W": stat, "p": p}
        except Exception as e:
            print(f"  {name_a} vs {name_b}: test failed ({e})")


def print_results_table(results: dict):
    """Print the main results table."""
    print("\n" + "="*75)
    print("MAIN RESULTS TABLE")
    print("="*75)
    print(f"{'Method':<25} {'ECE↓':>8} {'MCE↓':>8} {'Brier↓':>8} "
          f"{'Spear-r↑':>10} {'Hall%↓':>8}")
    print("-"*75)

    for name, m in results.items():
        if not isinstance(m, dict) or "ece" not in m:
            continue
        print(f"{name:<25} {m['ece']:>8.4f} {m['mce']:>8.4f} "
              f"{m['brier_score']:>8.4f} {m['spearman_r']:>10.4f} "
              f"{100*m['hallucination_rate']:>7.1f}%")

    print("="*75)
    print("↓ lower is better  ↑ higher is better")


def plot_calibration_hallucination_correlation(results: dict):
    """
    Scatter plot: ECE vs hallucination rate per method.
    The research claim: lower ECE = lower hallucination rate.
    """
    names  = []
    eces   = []
    hall_rates = []

    for name, m in results.items():
        if not isinstance(m, dict) or "ece" not in m:
            continue
        names.append(name)
        eces.append(m["ece"])
        hall_rates.append(m["hallucination_rate"])

    if len(names) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for name, ece, hr, color in zip(names, eces, hall_rates, colors):
        ax.scatter(ece, hr, s=150, color=color, zorder=5, label=name)
        ax.annotate(name, (ece, hr), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    # Trend line if enough points
    if len(eces) >= 3:
        z = np.polyfit(eces, hall_rates, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(eces) - 0.01, max(eces) + 0.01, 50)
        ax.plot(x_line, p(x_line), "k--", alpha=0.4, label="Trend")

    ax.set_xlabel("ECE (calibration error)")
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Calibration Error vs Hallucination Rate\n"
                 "(lower-left = better calibrated AND more faithful)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = os.path.join(FIGURES_DIR, "calibration_vs_hallucination.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Correlation plot saved → {out}")
    plt.show()
