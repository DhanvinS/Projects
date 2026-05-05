"""
evaluate.py
-----------
Computes FID between generated samples and CIFAR-10 real images.
Uses pytorch-fid under the hood.

DDPM sampling (reverse process):
    x_T ~ N(0, I)
    for t = T-1 down to 0:
        x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_phi(x_t, t))
                  + sigma_t * z    where z ~ N(0,I) if t > 0 else 0
"""

import torch
import numpy as np
import os
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
import config


def ddpm_sample(model, alpha_bar, device, n_samples=16, img_size=32, channels=3):
    """
    Generate n_samples images using DDPM reverse process.
    Returns tensor of shape (n_samples, C, H, W) in [-1, 1].
    """
    T = config.T
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, channels, img_size, img_size, device=device)
        for t_idx in reversed(range(T)):
            t_tensor = torch.full((n_samples,), t_idx, device=device, dtype=torch.long)
            eps_pred = model(x, t_tensor)

            alpha_t    = alphas[t_idx]
            alpha_bar_t = alpha_bar[t_idx]
            beta_t     = betas[t_idx]

            # Posterior mean
            coef = beta_t / (1 - alpha_bar_t).sqrt()
            mean = (1.0 / alpha_t.sqrt()) * (x - coef * eps_pred)

            if t_idx > 0:
                sigma_t = beta_t.sqrt()
                z = torch.randn_like(x)
                x = mean + sigma_t * z
            else:
                x = mean

    model.train()
    return x.clamp(-1, 1)


def save_images_to_dir(images, path):
    """Save (N, C, H, W) tensor in [-1,1] as PNGs to path/."""
    os.makedirs(path, exist_ok=True)
    images = (images * 0.5 + 0.5).clamp(0, 1)  # back to [0,1]
    for i, img in enumerate(images):
        torchvision.utils.save_image(img, os.path.join(path, f"{i:05d}.png"))


def save_real_images_to_dir(path, n=1000):
    """Save n real CIFAR-10 images to path/ for FID reference."""
    if os.path.exists(path) and len(os.listdir(path)) >= n:
        return  # already done
    os.makedirs(path, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    root = os.environ.get("CIFAR10_ROOT", "./data")
    if os.environ.get("CIFAR10_URL"):
        datasets.CIFAR10.url = os.environ["CIFAR10_URL"]
    download = os.environ.get("CIFAR10_DOWNLOAD", "1") != "0"
    dataset = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    saved = 0
    for imgs, _ in loader:
        for img in imgs:
            if saved >= n:
                break
            torchvision.utils.save_image(img, os.path.join(path, f"{saved:05d}.png"))
            saved += 1
        if saved >= n:
            break


def compute_fid(model, alpha_bar, device, n_samples=config.FID_SAMPLES):
    """Generate samples, save to disk, compute FID against real CIFAR-10."""
    gen_path  = "fid_tmp/generated"
    real_path = "fid_tmp/real"

    # Save real images once
    save_real_images_to_dir(real_path, n=n_samples)

    # Generate and save fake images
    batch_size = 64
    all_imgs = []
    for _ in range(0, n_samples, batch_size):
        imgs = ddpm_sample(model, alpha_bar, device,
                           n_samples=min(batch_size, n_samples - len(all_imgs)))
        all_imgs.append(imgs.cpu())
    all_imgs = torch.cat(all_imgs, dim=0)[:n_samples]
    save_images_to_dir(all_imgs, gen_path)

    # Compute FID
    fid = fid_score.calculate_fid_given_paths(
        [real_path, gen_path],
        batch_size=64,
        device=str(device),
        dims=2048
    )
    return fid
