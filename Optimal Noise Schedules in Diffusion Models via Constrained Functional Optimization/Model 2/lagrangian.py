"""
lagrangian.py
-------------
Augmented Lagrangian for the constrained theta optimisation.

Constraints:
    1. Coverage floor:  p(t; theta) >= epsilon   for all t
    2. Entropy floor:   H(p(theta)) >= H_min

Lagrangian:
    L(theta, lambda, mu) =  E[loss]
                          + lambda * sum_t  max(0, epsilon - p(t))^2
                          - mu     * (H(p(theta)) - H_min)

Dual updates (every N steps, gradient ascent on multipliers):
    lambda <- max(0, lambda + lr * coverage_violation)
    mu     <- max(0, mu     + lr * entropy_violation)
"""

import torch
import config


class AugmentedLagrangian:
    def __init__(self):
        self.lam = 0.0    # coverage multiplier (lambda)
        self.mu  = 0.0    # entropy multiplier

    def loss(self, theta_dist, expected_denoiser_loss):
        """
        Compute the full Lagrangian value.

        expected_denoiser_loss: scalar tensor (already IS-weighted mean over batch)
        Returns: scalar tensor (the thing theta is minimised over)
        """
        p = theta_dist.probs()              # (T,)
        eps = config.EPSILON

        # Coverage penalty: sum_t max(0, eps - p(t))^2
        coverage_violation = torch.clamp(eps - p, min=0.0).pow(2).sum()

        # Entropy penalty
        H = theta_dist.entropy()
        H_min = theta_dist.h_min()
        entropy_violation = H_min - H      # positive when entropy is too low

        lagrangian = (
            expected_denoiser_loss
            + self.lam * coverage_violation
            - self.mu  * (H - H_min)
        )
        return lagrangian, coverage_violation.item(), entropy_violation.item()

    def dual_update(self, coverage_violation, entropy_violation):
        """
        Gradient ascent on multipliers.
        coverage_violation: sum_t max(0, eps - p(t))^2  (scalar float)
        entropy_violation:  H_min - H(p)                (scalar float)
        """
        self.lam = max(0.0, self.lam + config.LAMBDA_LR * coverage_violation)
        self.mu  = max(0.0, self.mu  + config.MU_LR     * entropy_violation)
