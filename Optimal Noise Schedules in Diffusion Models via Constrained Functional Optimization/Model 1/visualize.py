"""
visualize.py
------------
Two main plots:

1. Curriculum heatmap:
   - x-axis: training step
   - y-axis: timestep t
   - colour:  p(t) value
   Shows how the learned distribution shifts across training.

2. Loss / FID curve:
   - phi loss over steps
   - FID over steps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # no display needed on server


def save_heatmap(p_history, path="logs/heatmap.png"):
    """
    p_history: list of dicts {"step": int, "p": list of floats (T,)}
    """
    if not p_history:
        return

    steps = [h["step"] for h in p_history]
    T = len(p_history[0]["p"])
    matrix = np.array([h["p"] for h in p_history]).T  # (T, num_snapshots)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
        extent=[steps[0], steps[-1], 0, T]
    )
    plt.colorbar(im, ax=ax, label="p(t; θ)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Timestep t")
    ax.set_title("Learned timestep distribution p(t; θ) across training")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_loss_curve(log, path="logs/loss_curve.png"):
    """
    log: dict with keys step, phi_loss, fid, fid_step
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Phi loss
    ax = axes[0]
    ax.plot(log["step"], log["phi_loss"], linewidth=1, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("IS-weighted MSE loss")
    ax.set_title("Denoiser loss (phi)")
    ax.grid(True, alpha=0.3)

    # FID
    ax = axes[1]
    if log["fid"]:
        ax.plot(log["fid_step"], log["fid"], marker="o", color="coral")
        ax.set_xlabel("Step")
        ax.set_ylabel("FID")
        ax.set_title("FID over training")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison_plot(results_dict, path="logs/comparison.png"):
    """
    Compare FID curves across methods.
    results_dict: {"uniform": {"steps": [...], "fid": [...]},
                   "min_snr": {...},
                   "learned": {...}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"uniform": "gray", "min_snr": "steelblue", "learned": "coral"}
    labels = {"uniform": "Uniform sampling", "min_snr": "Min-SNR (baseline)", "learned": "Learned p(t; θ)"}

    for key, data in results_dict.items():
        ax.plot(data["steps"], data["fid"],
                label=labels.get(key, key),
                color=colors.get(key, "black"),
                marker="o", linewidth=2)

    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("FID")
    ax.set_title("Compute efficiency: FID vs gradient steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
