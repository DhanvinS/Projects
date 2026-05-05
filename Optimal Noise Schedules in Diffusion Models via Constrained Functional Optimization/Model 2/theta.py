"""
theta.py
--------
Manages the learnable timestep sampling distribution p(t; theta).

theta: nn.Parameter of shape (T,) — one real number per timestep.
The distribution is obtained by softmax(theta).

Also provides:
- sample_timesteps(): draw a batch of timesteps from p(t; theta)
- importance_weights(): IS correction so denoiser gradient stays unbiased
- entropy(): H(p(theta)) for the entropy constraint
"""

import torch
import torch.nn as nn
import config


class ThetaDist(nn.Module):
    def __init__(self, T=config.T):
        super().__init__()
        self.T = T
        # initialise theta to zeros -> uniform distribution at start
        self.theta = nn.Parameter(torch.zeros(T))

    def probs(self):
        """Return p(t; theta) as a (T,) tensor. Differentiable."""
        return torch.softmax(self.theta, dim=0)

    def sample_timesteps(self, batch_size, device):
        """
        Sample batch_size timesteps from p(t; theta).
        Returns integer tensor of shape (batch_size,).
        """
        p = self.probs().detach()   # detach: sampling is not differentiable anyway
        t = torch.multinomial(p, num_samples=batch_size, replacement=True)
        return t.to(device)

    def importance_weights(self, t):
        """
        IS weight for each sampled timestep to keep denoiser gradient unbiased.
        w(t) = (1/T) / p(t)

        t: (B,) integer timestep indices
        Returns: (B,) weights
        """
        p = self.probs()
        p_t = p[t]                     # (B,)
        uniform = 1.0 / self.T
        w = uniform / p_t.detach()     # detach from theta graph; only phi is trained here
        return w

    def entropy(self):
        """Shannon entropy H(p(theta)) in nats."""
        p = self.probs()
        return -(p * (p + 1e-8).log()).sum()

    def h_min(self):
        """Entropy floor = H_MIN_FRACTION * log(T)."""
        import math
        return config.H_MIN_FRACTION * math.log(self.T)
