"""
Calibration Metrics

Three metrics — using all three is more credible than one:

1. ECE (Expected Calibration Error)
   - Bins predictions by confidence, measures mean |confidence - accuracy| per bin
   - Standard calibration metric (Guo et al. 2017, Naeini et al. 2015)
   - Weighted by bin size — large bins matter more

2. MCE (Maximum Calibration Error)
   - Worst-case calibration across bins
   - ECE can hide severe miscalibration in small bins — MCE catches it
   - Important for high-stakes applications (a model that's wildly wrong
     10% of the time matters even if average calibration is fine)

3. Brier Score
   - Mean squared error between confidence and binary outcome
   - Proper scoring rule — cannot be gamed by arbitrary recalibration
   - Decomposes into calibration + refinement components

The key research claim:
  Does uncertainty-aware decoding (Method B) achieve lower ECE/MCE/Brier
  than post-hoc temperature scaling (Method A)?
  And does lower ECE correlate with fewer hallucinations?
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def compute_ece(
    confidences: list[float],
    labels: list[int],
    n_bins: int = 10,
) -> dict:
    """
    Expected Calibration Error.
    confidences: model confidence per sentence [0,1]
    labels: 1 = factually consistent, 0 = hallucinated
    """
    confidences = np.array(confidences)
    labels      = np.array(labels)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    bin_stats = []

    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            bin_stats.append({"bin": b, "count": 0, "confidence": 0, "accuracy": 0})
            continue

        bin_conf = confidences[mask].mean()
        bin_acc  = labels[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / len(confidences)) * abs(bin_conf - bin_acc)

        bin_stats.append({
            "bin": b,
            "count": int(bin_size),
            "confidence": float(bin_conf),
            "accuracy": float(bin_acc),
            "gap": float(abs(bin_conf - bin_acc)),
        })

    return {"ece": float(ece), "bins": bin_stats, "n_bins": n_bins}


def compute_mce(ece_result: dict) -> float:
    """
    Maximum Calibration Error — worst-case bin gap.
    Computed from ECE result dict to avoid duplicate binning.
    """
    gaps = [b["gap"] for b in ece_result["bins"] if b["count"] > 0]
    return float(max(gaps)) if gaps else 0.0


def compute_brier_score(
    confidences: list[float],
    labels: list[int],
) -> float:
    """
    Brier Score = mean((confidence - label)^2).
    Lower is better. Perfect calibration = 0, random = 0.25.
    """
    confidences = np.array(confidences)
    labels      = np.array(labels)
    return float(np.mean((confidences - labels) ** 2))


def compute_all_metrics(
    confidences: list[float],
    labels: list[int],
    method_name: str = "",
) -> dict:
    """Compute ECE, MCE, and Brier Score in one call."""
    ece_result = compute_ece(confidences, labels)
    mce        = compute_mce(ece_result)
    brier      = compute_brier_score(confidences, labels)

    results = {
        "method":      method_name,
        "ece":         ece_result["ece"],
        "mce":         mce,
        "brier_score": brier,
        "n_samples":   len(confidences),
        "bin_stats":   ece_result["bins"],
    }

    print(f"\n── Calibration Metrics: {method_name} ──")
    print(f"  ECE:         {results['ece']:.4f}  (lower = better)")
    print(f"  MCE:         {results['mce']:.4f}  (lower = better)")
    print(f"  Brier Score: {results['brier_score']:.4f}  (lower = better)")
    print(f"  N samples:   {results['n_samples']}")

    return results


def plot_reliability_diagram(
    results_list: list[dict],
    output_path: str = "experiments/figures/reliability_diagram.png",
):
    """
    Reliability diagram for multiple methods on one plot.
    Perfect calibration = diagonal line.
    Points above diagonal = underconfident.
    Points below diagonal = overconfident.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, axes = plt.subplots(1, len(results_list), figsize=(5 * len(results_list), 5))

    if len(results_list) == 1:
        axes = [axes]

    for ax, result, color in zip(axes, results_list, colors):
        bins = result["bin_stats"]
        bin_confs = [b["confidence"] for b in bins if b["count"] > 0]
        bin_accs  = [b["accuracy"]   for b in bins if b["count"] > 0]
        bin_sizes = [b["count"]      for b in bins if b["count"] > 0]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        # Calibration curve — bubble size = bin size
        scatter = ax.scatter(
            bin_confs, bin_accs,
            s=[s * 2 for s in bin_sizes],
            c=color, alpha=0.7, zorder=5,
        )
        ax.plot(bin_confs, bin_accs, color=color, alpha=0.6)

        # Gap fill (overconfidence = red, underconfidence = blue)
        for bc, ba in zip(bin_confs, bin_accs):
            color_fill = "#FF000033" if bc > ba else "#0000FF22"
            ax.fill_between([bc - 0.05, bc + 0.05], [bc, bc], [ba, ba],
                            alpha=0.3, color=color_fill)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy (Factual Consistency)")
        ax.set_title(
            f"{result['method']}\n"
            f"ECE={result['ece']:.3f}  MCE={result['mce']:.3f}  "
            f"Brier={result['brier_score']:.3f}"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Reliability Diagrams: Calibration Comparison", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Reliability diagram saved → {output_path}")
    plt.show()
