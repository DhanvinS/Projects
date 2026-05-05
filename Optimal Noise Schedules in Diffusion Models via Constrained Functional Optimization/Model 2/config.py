import torch

T = 100                   # number of diffusion timesteps
IMG_SIZE = 32             # CIFAR-10 image size
CHANNELS = 3              # RGB
BATCH_SIZE = 128
LR_PHI = 2e-4             # denoiser learning rate
LR_THETA = 1e-2           # theta learning rate
THETA_UPDATE_EVERY = 500  # update theta every N denoiser steps
TOTAL_STEPS = 100_000     # total denoiser gradient steps
THETA_ESTIMATE_BATCHES = 8 # batches used to estimate per-timestep loss
THETA_LOSS_EMA = 0.9      # smooth per-timestep loss estimates for learned samplers
THETA_LOSS_POWER = 0.5    # target p(t) proportional to loss^power for loss-weighted mode
HYBRID_UNIFORM_MIX = 0.7  # learned_hybrid uses this much uniform sampling
HYBRID_PROB_FLOOR = 0.002 # hard minimum probability per timestep before uniform mixing

# Lagrangian / constraints
EPSILON = 0.002           # coverage floor: p(t) >= epsilon for all t
H_MIN_FRACTION = 0.5      # entropy floor: H(p) >= H_MIN_FRACTION * log(T)
LAMBDA_LR = 1e-3          # dual ascent step for lambda (coverage)
MU_LR = 1e-3              # dual ascent step for mu (entropy)

# Logging / eval
LOG_EVERY = 500
EVAL_EVERY = 5000
FID_SAMPLES = 1000        # number of samples to compute FID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
