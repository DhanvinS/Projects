PyTorch, Diffusion Models, Lagrangian Optimization, Adam, Cubic Splines
• Reframed noise schedule design (βt) as a constrained optimization problem — treating it as a learnable differentiable function
(quadratic, cubic spline, MLP) — and minimized diffusion training loss subject to smoothness and stability constraints via
Lagrangian penalty methods.
• Evaluated three parameterizations on MNIST (T=100) with Adam/SGD/L-BFGS; gradient flow analysis showed cubic splines offer the
best expressiveness-stability tradeoff over linear and cosine baselines.
