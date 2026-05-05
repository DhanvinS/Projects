"""
diffusion.py
------------
Implements the DDPM forward process (noising) and the denoising loss.

Forward process:
    q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

Loss (per sample):
    L(t) = || eps_phi(x_t, t) - eps ||^2
    where eps ~ N(0, I) is the noise added, x_t is the noisy image.
"""

import torch
import torch.nn.functional as F
import config


def make_noise_schedule(T, device):
    """
    Linear beta schedule from DDPM paper.
    Returns alpha_bar: cumulative product of (1 - beta_t), shape (T,)
    """
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)   # (T,)
    return alpha_bar


def q_sample(x0, t, alpha_bar, noise=None):
    """
    Sample x_t from the forward process given x_0 and timestep t.

    x0:        (B, C, H, W)
    t:         (B,) integer timesteps in [0, T-1]
    alpha_bar: (T,) precomputed cumulative alphas
    Returns:   x_t (B, C, H, W), noise (B, C, H, W)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    ab = alpha_bar[t][:, None, None, None]          # (B,1,1,1)
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    return x_t, noise


def compute_loss_per_sample(model, x0, t, alpha_bar):
    """
    Compute per-sample MSE loss for a batch.

    Returns:
        loss_per_sample: (B,) — one scalar loss per sample
    """
    x_t, noise = q_sample(x0, t, alpha_bar)
    pred_noise = model(x_t, t)
    # mean over pixel dims, keep batch dim
    loss = F.mse_loss(pred_noise, noise, reduction='none')
    loss_per_sample = loss.mean(dim=[1, 2, 3])   # (B,)
    return loss_per_sample
