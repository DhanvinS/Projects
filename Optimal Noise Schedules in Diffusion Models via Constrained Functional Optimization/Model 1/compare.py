"""
compare.py
----------
Loads saved logs from all runs and produces:
  1. FID vs gradient steps comparison plot (3 main conditions)
  2. Ablation table (printed + saved as CSV)
  3. Side-by-side curriculum heatmaps

Run this after all training runs are complete:
    python compare.py

Expected folder structure (one per mode):
    runs/uniform/logs/log.json
    runs/minsnr/logs/log.json
    runs/learned/logs/log.json
    runs/learned_no_constraints/logs/log.json
    runs/fixed_subset/logs/log.json
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv


# ─────────────────────────────────────────
# Load logs
# ─────────────────────────────────────────

ALL_MODES = [
    "uniform",
    "minsnr",
    "learned",
    "learned_no_constraints",
    "fixed_subset",
]

LABELS = {
    "uniform":                "Uniform sampling",
    "minsnr":                 "Min-SNR (baseline)",
    "learned":                "Learned p(t; θ) — full",
    "learned_no_constraints": "Learned p(t; θ) — no constraints",
    "fixed_subset":           "Fixed subset (t=60–99)",
}

COLORS = {
    "uniform":                "#888888",
    "minsnr":                 "#4C72B0",
    "learned":                "#DD4444",
    "learned_no_constraints": "#EE9944",
    "fixed_subset":           "#44AA88",
}


def load_logs():
    logs = {}
    for mode in ALL_MODES:
        path = f"runs/{mode}/logs/log.json"
        if os.path.exists(path):
            with open(path) as f:
                logs[mode] = json.load(f)
            print(f"  Loaded: {mode}")
        else:
            print(f"  Missing: {mode}  (skipping)")
    return logs


# ─────────────────────────────────────────
# Plot 1: FID vs gradient steps
# ─────────────────────────────────────────

def plot_fid_curves(logs, path="results/fid_comparison.png"):
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    for mode, log in logs.items():
        if not log.get("fid"):
            continue
        ax.plot(
            log["fid_step"], log["fid"],
            label=LABELS[mode],
            color=COLORS[mode],
            marker="o", linewidth=2, markersize=5
        )

    ax.set_xlabel("Gradient steps", fontsize=12)
    ax.set_ylabel("FID ↓", fontsize=12)
    ax.set_title("Compute efficiency: FID vs gradient steps (CIFAR-10)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# Plot 2: Ablation table
# ─────────────────────────────────────────

def steps_to_target_fid(fid_list, step_list, target=50.0):
    """Return the first step where FID drops below target, or None."""
    for fid, step in zip(fid_list, step_list):
        if fid <= target:
            return step
    return None


def print_ablation_table(logs, target_fid=50.0, csv_path="results/ablation.csv"):
    os.makedirs("results", exist_ok=True)

    rows = []
    for mode, log in logs.items():
        fid_list  = log.get("fid", [])
        step_list = log.get("fid_step", [])
        final_fid = fid_list[-1] if fid_list else float("nan")
        best_fid  = min(fid_list) if fid_list else float("nan")
        steps_to_target = steps_to_target_fid(fid_list, step_list, target=target_fid)
        rows.append({
            "Mode":             LABELS[mode],
            "Final FID":        f"{final_fid:.2f}",
            "Best FID":         f"{best_fid:.2f}",
            f"Steps to FID<{target_fid}": steps_to_target if steps_to_target else "Not reached",
        })

    # Print to terminal
    col_widths = {k: max(len(k), max(len(str(r[k])) for r in rows)) for k in rows[0]}
    header = "  ".join(k.ljust(col_widths[k]) for k in col_widths)
    print("\n" + "─" * len(header))
    print("ABLATION TABLE")
    print("─" * len(header))
    print(header)
    print("─" * len(header))
    for row in rows:
        print("  ".join(str(row[k]).ljust(col_widths[k]) for k in col_widths))
    print("─" * len(header) + "\n")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_path}")


# ─────────────────────────────────────────
# Plot 3: Side-by-side heatmaps
# ─────────────────────────────────────────

def plot_heatmaps(logs, path="results/heatmaps.png"):
    """
    Show curriculum heatmap for each mode side by side.
    Only modes that have p_history are included.
    """
    os.makedirs("results", exist_ok=True)

    modes_with_history = [m for m in logs if logs[m].get("p_history")]
    n = len(modes_with_history)
    if n == 0:
        print("  No p_history found, skipping heatmaps.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes_with_history):
        history = logs[mode]["p_history"]
        steps   = [h["step"] for h in history]
        T       = len(history[0]["p"])
        matrix  = np.array([h["p"] for h in history]).T   # (T, snapshots)

        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            interpolation="nearest",
            extent=[steps[0], steps[-1], 0, T],
            vmin=0
        )
        ax.set_title(LABELS[mode], fontsize=9)
        ax.set_xlabel("Step", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Timestep t", fontsize=8)

    fig.colorbar(im, ax=axes[-1], label="p(t; θ)")
    fig.suptitle("Learned distribution p(t) across training", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# Plot 4: Per-timestep loss evolution
# ─────────────────────────────────────────

def plot_per_timestep_loss(logs, path="results/per_t_loss.png"):
    """
    For the learned mode, show how per-timestep loss changes across training.
    This directly shows which timesteps remain hard.
    """
    os.makedirs("results", exist_ok=True)

    if "learned" not in logs or not logs["learned"].get("per_t_loss_history"):
        print("  No per_t_loss_history for learned mode, skipping.")
        return

    history = logs["learned"]["per_t_loss_history"]
    steps   = [h["step"] for h in history]
    T       = len(history[0]["loss"])
    matrix  = np.array([h["loss"] for h in history]).T   # (T, snapshots)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        interpolation="nearest",
        extent=[steps[0], steps[-1], 0, T]
    )
    plt.colorbar(im, ax=ax, label="MSE loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Timestep t")
    ax.set_title("Per-timestep denoiser loss across training (learned mode)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\nLoading logs...")
    logs = load_logs()

    if not logs:
        print("No logs found. Run train.py first.")
        exit()

    print("\nGenerating plots and table...")
    plot_fid_curves(logs)
    print_ablation_table(logs, target_fid=50.0)
    plot_heatmaps(logs)
    plot_per_timestep_loss(logs)

    print("\nAll results saved to results/")
