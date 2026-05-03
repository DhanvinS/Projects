"""
Small UNet denoiser (phi).
Input:  noisy image x_t (B, C, H, W) + timestep t (B,)
Output: predicted noise eps (B, C, H, W)

Architecture: encoder-decoder with skip connections and sinusoidal time embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- time embedding ----------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) integer timesteps
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return embedding


# ---------- building blocks ----------

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        return self.down(x), x   # return downsampled + skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res = ResBlock(in_ch + skip_ch, out_ch, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x, t_emb)


# ---------- UNet ----------

class UNet(nn.Module):
    """
    Channels: 32 -> 64 -> 128 -> 128 (bottleneck) -> 128 -> 64 -> 32
    Works for 32x32 CIFAR images.
    """
    def __init__(self, in_channels=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        self.init_conv = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch,      base_ch * 2, time_dim)  # 32->16
        self.down2 = DownBlock(base_ch * 2,  base_ch * 4, time_dim)  # 16->8

        self.mid1 = ResBlock(base_ch * 4, base_ch * 4, time_dim)
        self.mid2 = ResBlock(base_ch * 4, base_ch * 4, time_dim)

        self.up1 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, time_dim)  # 8->16
        self.up2 = UpBlock(base_ch * 2, base_ch * 2, base_ch,     time_dim)  # 16->32

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_emb(t))

        x = self.init_conv(x)

        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)

        return self.out_conv(F.silu(self.out_norm(x)))
