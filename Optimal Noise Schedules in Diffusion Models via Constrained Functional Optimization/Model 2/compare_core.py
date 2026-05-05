"""
compare_core.py
---------------
Focused comparison for the three core runs:
    uniform, learned, learned_hybrid

Run after learned_hybrid finishes:
    python compare_core.py

Outputs:
    results_core/summary.csv
    results_core/summary.json
    results_core/fid_curves.png
    results_core/best_fid_curves.png
    results_core/loss_curves.png
    results_core/sampler_heatmaps.png
    results_core/final_probabilities.png
    results_core/per_timestep_loss_heatmaps.png
"""

import csv
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODES = ["uniform", "learned", "learned_hybrid"]
FID_PLOT_MODES = ["uniform", "learned_hybrid"]

LABELS = {
    "uniform": "Uniform",
    "learned": "Learned",
    "learned_hybrid": "Hybrid learned",
}

COLORS = {
    "uniform": "#6F6F6F",
    "learned": "#D64B4B",
    "learned_hybrid": "#188F8C",
}

OUT_DIR = "results_core"


def load_logs():
    logs = {}
    for mode in MODES:
        path = os.path.join("runs", mode, "logs", "log.json")
        if not os.path.exists(path):
            print(f"Missing {mode}: {path}")
            continue
        with open(path, "r") as f:
            logs[mode] = json.load(f)
        print(f"Loaded {mode}: {path}")
    return logs


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def finite_or_none(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def area_under_curve(x, y):
    if len(x) < 2:
        return float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def steps_to_best_so_far(fid_steps, fids, targets):
    out = {}
    if len(fids) == 0:
        return {f"steps_to_fid_le_{target}": None for target in targets}

    best_so_far = np.minimum.accumulate(fids)
    for target in targets:
        key = f"steps_to_fid_le_{target}"
        hit = np.where(best_so_far <= target)[0]
        out[key] = int(fid_steps[hit[0]]) if len(hit) else None
    return out


def summarize_mode(mode, log):
    fid_steps = np.array(log.get("fid_step", []), dtype=float)
    fids = np.array(log.get("fid", []), dtype=float)
    loss_steps = np.array(log.get("step", []), dtype=float)
    losses = np.array(log.get("phi_loss", []), dtype=float)

    summary = {
        "mode": mode,
        "label": LABELS[mode],
        "complete": int(loss_steps[-1]) >= 100000 if len(loss_steps) else False,
        "latest_step": int(loss_steps[-1]) if len(loss_steps) else None,
        "fid_evals": int(len(fids)),
        "first_fid": float(fids[0]) if len(fids) else None,
        "latest_fid": float(fids[-1]) if len(fids) else None,
        "latest_fid_step": int(fid_steps[-1]) if len(fid_steps) else None,
        "best_fid": None,
        "best_fid_step": None,
        "fid_auc": None,
        "first_loss": float(losses[0]) if len(losses) else None,
        "latest_loss": float(losses[-1]) if len(losses) else None,
        "best_logged_loss": float(losses.min()) if len(losses) else None,
    }

    if len(fids):
        best_idx = int(np.argmin(fids))
        summary["best_fid"] = float(fids[best_idx])
        summary["best_fid_step"] = int(fid_steps[best_idx])
        summary["fid_auc"] = area_under_curve(fid_steps, fids)
        summary.update(steps_to_best_so_far(fid_steps, fids, targets=[175, 160, 150, 140]))
    else:
        summary.update(steps_to_best_so_far(fid_steps, fids, targets=[175, 160, 150, 140]))

    p_history = log.get("p_history", [])
    if p_history:
        p = np.array(p_history[-1]["p"], dtype=float)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        top = sorted([(float(v), int(i)) for i, v in enumerate(p)], reverse=True)[:10]
        bottom = sorted([(float(v), int(i)) for i, v in enumerate(p)])[:10]
        summary.update({
            "p_step": int(p_history[-1]["step"]),
            "p_min": float(p.min()),
            "p_max": float(p.max()),
            "p_std": float(p.std()),
            "p_entropy": entropy,
            "p_top10": "; ".join(f"t={i}:{v:.5f}" for v, i in top),
            "p_bottom10": "; ".join(f"t={i}:{v:.5f}" for v, i in bottom),
        })

    return {k: finite_or_none(v) for k, v in summary.items()}


def write_summary(logs):
    summaries = [summarize_mode(mode, logs[mode]) for mode in MODES if mode in logs]
    json_path = os.path.join(OUT_DIR, "summary.json")
    csv_path = os.path.join(OUT_DIR, "summary.csv")

    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    fields = list(summaries[0].keys()) if summaries else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    print("\nSUMMARY")
    for row in summaries:
        print(
            f"{row['label']}: best FID={row['best_fid']:.2f} @ {row['best_fid_step']}, "
            f"latest FID={row['latest_fid']:.2f} @ {row['latest_fid_step']}"
        )
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")


def plot_fid_curves(logs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in FID_PLOT_MODES:
        if mode not in logs or not logs[mode].get("fid"):
            continue
        ax.plot(
            logs[mode]["fid_step"],
            logs[mode]["fid"],
            marker="o",
            linewidth=2,
            markersize=4,
            label=LABELS[mode],
            color=COLORS[mode],
        )
    ax.set_title("FID vs Training Steps")
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("FID lower is better")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "fid_curves.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def plot_best_fid_curves(logs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in MODES:
        if mode not in logs or not logs[mode].get("fid"):
            continue
        steps = np.array(logs[mode]["fid_step"])
        fids = np.array(logs[mode]["fid"])
        ax.plot(
            steps,
            np.minimum.accumulate(fids),
            marker="o",
            linewidth=2,
            markersize=4,
            label=LABELS[mode],
            color=COLORS[mode],
        )
    ax.set_title("Best FID Reached So Far")
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("Best FID so far lower is better")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "best_fid_curves.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def plot_loss_curves(logs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in MODES:
        if mode not in logs or not logs[mode].get("phi_loss"):
            continue
        ax.plot(
            logs[mode]["step"],
            logs[mode]["phi_loss"],
            linewidth=1.6,
            label=LABELS[mode],
            color=COLORS[mode],
        )
    ax.set_title("Training Loss")
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("IS-weighted denoising MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "loss_curves.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def plot_sampler_heatmaps(logs):
    modes = [m for m in MODES if m in logs and logs[m].get("p_history")]
    if not modes:
        return

    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        history = logs[mode]["p_history"]
        steps = [h["step"] for h in history]
        matrix = np.array([h["p"] for h in history]).T
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="inferno",
            interpolation="nearest",
            extent=[steps[0], steps[-1], 0, matrix.shape[0]],
        )
        ax.set_title(LABELS[mode])
        ax.set_xlabel("Gradient steps")
        if ax == axes[0]:
            ax.set_ylabel("Timestep")
    fig.colorbar(im, ax=axes, label="Sampling probability")
    path = os.path.join(OUT_DIR, "sampler_heatmaps.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_final_probabilities(logs):
    fig, ax = plt.subplots(figsize=(9, 5))
    for mode in MODES:
        if mode not in logs or not logs[mode].get("p_history"):
            continue
        p = logs[mode]["p_history"][-1]["p"]
        ax.plot(
            range(len(p)),
            p,
            linewidth=2,
            label=LABELS[mode],
            color=COLORS[mode],
        )
    ax.set_title("Final Timestep Sampling Distribution")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("p(t)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "final_probabilities.png")
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Saved {path}")


def plot_per_timestep_loss_heatmaps(logs):
    modes = [m for m in MODES if m in logs and logs[m].get("per_t_loss_history")]
    if not modes:
        return

    fig, axes = plt.subplots(1, len(modes), figsize=(5 * len(modes), 4), sharey=True)
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        history = logs[mode]["per_t_loss_history"]
        steps = [h["step"] for h in history]
        matrix = np.array([h["loss"] for h in history]).T
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            extent=[steps[0], steps[-1], 0, matrix.shape[0]],
        )
        ax.set_title(LABELS[mode])
        ax.set_xlabel("Gradient steps")
        if ax == axes[0]:
            ax.set_ylabel("Timestep")
    fig.colorbar(im, ax=axes, label="Per-timestep MSE")
    path = os.path.join(OUT_DIR, "per_timestep_loss_heatmaps.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    ensure_out_dir()
    logs = load_logs()
    if not logs:
        print("No logs found.")
        return

    write_summary(logs)
    plot_fid_curves(logs)
    plot_best_fid_curves(logs)
    plot_loss_curves(logs)
    plot_sampler_heatmaps(logs)
    plot_final_probabilities(logs)
    plot_per_timestep_loss_heatmaps(logs)
    print(f"\nAll core comparison results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
