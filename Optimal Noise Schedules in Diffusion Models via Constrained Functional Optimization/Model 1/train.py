"""
train.py
--------
Main training loop with mode flag to run all conditions.

Usage:
    python train.py --mode uniform
    python train.py --mode minsnr
    python train.py --mode learned
    python train.py --mode learned_no_constraints
    python train.py --mode fixed_subset --subset_start 60 --subset_end 100

Modes:
    uniform               - standard DDPM, t ~ Uniform{1,...,T}
    minsnr                - fixed IS weights by Min-SNR schedule (gamma=5)
    learned               - full method: learned p(t;theta) + both constraints
    learned_no_constraints- learned p(t;theta) with no coverage or entropy floor
    fixed_subset          - pin uniform distribution over a fixed range of timesteps
"""

import torch
import torch.optim as optim
import numpy as np
import os
import json
import argparse
from itertools import cycle

import config
from dataset import get_cifar10_loader
from unet import UNet
from diffusion import make_noise_schedule, compute_loss_per_sample
from theta import ThetaDist
from lagrangian import AugmentedLagrangian
from evaluate import compute_fid
from visualize import save_heatmap, save_loss_curve


# ─────────────────────────────────────────
# Min-SNR weights (Hang et al. 2023)
# ─────────────────────────────────────────

def make_minsnr_weights(alpha_bar, gamma=5):
    """
    Min-SNR weighting: w(t) = min(SNR_t, gamma) / SNR_t
    SNR_t = alpha_bar_t / (1 - alpha_bar_t)
    Returns (T,) tensor normalised to sum to 1.
    """
    snr = alpha_bar / (1.0 - alpha_bar + 1e-8)
    w = torch.clamp(snr, max=gamma) / (snr + 1e-8)
    w = w / w.sum()
    return w


# ─────────────────────────────────────────
# Fixed subset distribution
# ─────────────────────────────────────────

def make_fixed_subset_dist(T, start, end, device):
    """Uniform over timesteps [start, end] inclusive, zero elsewhere."""
    p = torch.zeros(T, device=device)
    p[start:end + 1] = 1.0
    p = p / p.sum()
    return p


# ─────────────────────────────────────────
# Per-timestep loss estimator
# ─────────────────────────────────────────

def compute_per_timestep_loss(model, loader_iter, alpha_bar, device, n_batches=4):
    T = config.T
    loss_accum  = np.zeros(T)
    count_accum = np.zeros(T) + 1e-8

    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            x0, _ = next(loader_iter)
            x0 = x0.to(device)
            B  = x0.shape[0]
            t  = torch.randint(0, T, (B,), device=device)
            loss = compute_loss_per_sample(model, x0, t, alpha_bar)
            for i in range(B):
                ti = t[i].item()
                loss_accum[ti]  += loss[i].item()
                count_accum[ti] += 1

    model.train()
    return loss_accum / count_accum


# ─────────────────────────────────────────
# Sampler classes
# ─────────────────────────────────────────

class UniformSampler:
    """t ~ Uniform{0,...,T-1}, IS weight = 1 (already unbiased)."""
    def __init__(self, T, device):
        self.T = T
        self.device = device

    def sample(self, B):
        return torch.randint(0, self.T, (B,), device=self.device)

    def weights(self, t):
        return torch.ones(t.shape[0], device=self.device)

    def probs(self):
        return torch.full((self.T,), 1.0 / self.T, device=self.device)


class MinSNRSampler:
    """Sample from Min-SNR distribution with IS correction back to uniform."""
    def __init__(self, alpha_bar, device, gamma=5):
        self.device = device
        self.T = alpha_bar.shape[0]
        self.p = make_minsnr_weights(alpha_bar, gamma=gamma).to(device)

    def sample(self, B):
        return torch.multinomial(self.p, num_samples=B, replacement=True)

    def weights(self, t):
        uniform = 1.0 / self.T
        return (uniform / (self.p[t] + 1e-8)).detach()

    def probs(self):
        return self.p


class FixedSubsetSampler:
    """Uniform over a fixed range of timesteps, IS correction applied."""
    def __init__(self, T, start, end, device):
        self.device = device
        self.T = T
        self.p = make_fixed_subset_dist(T, start, end, device)

    def sample(self, B):
        return torch.multinomial(self.p, num_samples=B, replacement=True)

    def weights(self, t):
        uniform = 1.0 / self.T
        return (uniform / (self.p[t] + 1e-8)).detach()

    def probs(self):
        return self.p


class LearnedSampler:
    """Learned p(t; theta) with optional Lagrangian constraints."""
    def __init__(self, T, device, use_constraints=True):
        self.device = device
        self.T = T
        self.use_constraints = use_constraints
        self.theta_dist  = ThetaDist(T=T).to(device)
        self.theta_opt   = optim.Adam(self.theta_dist.parameters(), lr=config.LR_THETA)
        self.lagrangian  = AugmentedLagrangian()

    def sample(self, B):
        return self.theta_dist.sample_timesteps(B, self.device)

    def weights(self, t):
        return self.theta_dist.importance_weights(t)

    def probs(self):
        return self.theta_dist.probs()

    def update(self, per_t_loss_np):
        """Gradient step on theta + dual ascent on multipliers."""
        per_t_loss_tensor = torch.tensor(
            per_t_loss_np, dtype=torch.float32, device=self.device
        )
        p = self.theta_dist.probs()
        expected_loss = (p * per_t_loss_tensor).sum()

        if self.use_constraints:
            lag_val, cov_viol, ent_viol = self.lagrangian.loss(
                self.theta_dist, expected_loss
            )
        else:
            lag_val = expected_loss
            cov_viol, ent_viol = 0.0, 0.0

        self.theta_opt.zero_grad()
        lag_val.backward()
        self.theta_opt.step()

        if self.use_constraints:
            self.lagrangian.dual_update(cov_viol, ent_viol)

        return cov_viol, ent_viol


# ─────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────

def train(mode, subset_start=60, subset_end=99):
    device  = torch.device(config.DEVICE)
    run_dir = f"runs/{mode}"
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{run_dir}/logs", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Mode: {mode}")
    if mode == "fixed_subset":
        print(f"  Subset: t in [{subset_start}, {subset_end}]")
    print(f"{'='*50}\n")

    # Data
    loader      = get_cifar10_loader(config.BATCH_SIZE)
    loader_iter = cycle(loader)

    # Noise schedule
    alpha_bar = make_noise_schedule(config.T, device)

    # Denoiser
    model   = UNet(in_channels=config.CHANNELS).to(device)
    phi_opt = optim.Adam(model.parameters(), lr=config.LR_PHI)

    # Sampler
    if mode == "uniform":
        sampler    = UniformSampler(config.T, device)
        is_learned = False

    elif mode == "minsnr":
        sampler    = MinSNRSampler(alpha_bar, device, gamma=5)
        is_learned = False

    elif mode == "learned":
        sampler    = LearnedSampler(config.T, device, use_constraints=True)
        is_learned = True

    elif mode == "learned_no_constraints":
        sampler    = LearnedSampler(config.T, device, use_constraints=False)
        is_learned = True

    elif mode == "fixed_subset":
        sampler    = FixedSubsetSampler(config.T, subset_start, subset_end, device)
        is_learned = False

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Log
    log = {
        "mode": mode,
        "step": [],
        "phi_loss": [],
        "fid": [],
        "fid_step": [],
        "p_history": [],
        "per_t_loss_history": [],
        "lambda_history": [],
        "mu_history": [],
    }

    model.train()

    for step in range(1, config.TOTAL_STEPS + 1):

        # ── Phi update ──────────────────────────────
        x0, _ = next(loader_iter)
        x0 = x0.to(device)
        B  = x0.shape[0]

        t = sampler.sample(B)
        w = sampler.weights(t)

        loss_per_sample = compute_loss_per_sample(model, x0, t, alpha_bar)
        phi_loss = (w * loss_per_sample).mean()

        phi_opt.zero_grad()
        phi_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        phi_opt.step()

        # ── Theta update (learned modes only) ───────
        if step % config.THETA_UPDATE_EVERY == 0:
            per_t_loss = compute_per_timestep_loss(
                model, loader_iter, alpha_bar, device, n_batches=4
            )

            if is_learned:
                cov_viol, ent_viol = sampler.update(per_t_loss)
                with torch.no_grad():
                    H   = sampler.theta_dist.entropy().item()
                    lam = sampler.lagrangian.lam if sampler.use_constraints else 0.0
                    mu  = sampler.lagrangian.mu  if sampler.use_constraints else 0.0
                log["lambda_history"].append(lam)
                log["mu_history"].append(mu)
                print(
                    f"[{step:6d}] phi_loss={phi_loss.item():.4f} | "
                    f"H={H:.3f} | lambda={lam:.4f} | mu={mu:.4f}"
                )

            # Save p(t) snapshot for all modes
            with torch.no_grad():
                p_np = sampler.probs().cpu().numpy().tolist()
            log["p_history"].append({"step": step, "p": p_np})
            log["per_t_loss_history"].append({"step": step, "loss": per_t_loss.tolist()})

        # ── Scalar logging ───────────────────────────
        if step % config.LOG_EVERY == 0:
            log["step"].append(step)
            log["phi_loss"].append(phi_loss.item())
            if step % (config.LOG_EVERY * 5) == 0:
                print(f"[{step:6d}] phi_loss={phi_loss.item():.4f}")

        # ── FID eval ─────────────────────────────────
        if step % config.EVAL_EVERY == 0:
            fid = compute_fid(model, alpha_bar, device, n_samples=config.FID_SAMPLES)
            log["fid"].append(fid)
            log["fid_step"].append(step)
            print(f"[{step:6d}] FID = {fid:.2f}")

            torch.save(model.state_dict(),
                       f"{run_dir}/checkpoints/model_{step}.pt")
            if is_learned:
                torch.save(sampler.theta_dist.state_dict(),
                           f"{run_dir}/checkpoints/theta_{step}.pt")

            if log["p_history"]:
                save_heatmap(log["p_history"],
                             path=f"{run_dir}/logs/heatmap_{step}.png")
            save_loss_curve(log, path=f"{run_dir}/logs/loss_curve_{step}.png")

            with open(f"{run_dir}/logs/log.json", "w") as f:
                json.dump(log, f)

    # Final save
    with open(f"{run_dir}/logs/log.json", "w") as f:
        json.dump(log, f)
    print(f"\nDone. Logs saved to {run_dir}/logs/")


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="learned",
        choices=["uniform", "minsnr", "learned", "learned_no_constraints", "fixed_subset"]
    )
    parser.add_argument("--subset_start", type=int, default=60,
                        help="Start of fixed timestep subset (inclusive, 0-indexed)")
    parser.add_argument("--subset_end",   type=int, default=99,
                        help="End of fixed timestep subset (inclusive, 0-indexed)")
    args = parser.parse_args()

    train(
        mode=args.mode,
        subset_start=args.subset_start,
        subset_end=args.subset_end,
    )
