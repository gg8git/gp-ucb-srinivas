# Reimplementation of the experimental design described
# Experiments:
# 1) Sublinear regret comparison (GP-UCB vs EI vs Random)
# 2) Robustness to kernel lengthscale misspecification

import math
from tqdm import tqdm
import torch
import gpytorch
import numpy as np
from torch.distributions.normal import Normal

# -----------------------------
# GP model
# -----------------------------

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=lengthscale)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# -----------------------------
# Acquisition functions
# -----------------------------

def gp_ucb(model, likelihood, X, beta):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
        return pred.mean - math.sqrt(beta) * pred.variance.sqrt()


def expected_improvement(model, likelihood, X, y_best):
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = likelihood(model(X))
        mu = pred.mean
        sigma = pred.variance.sqrt().clamp_min(1e-9)
        Z = (y_best - mu) / sigma
        normal = Normal(0, 1)
        return (y_best - mu) * normal.cdf(Z) + sigma * normal.log_prob(Z).exp()

# -----------------------------
# Utilities
# -----------------------------

def sample_uniform(n, bounds):
    u = torch.rand(n, 2).to("cpu")
    return bounds[0] + (bounds[1] - bounds[0]) * u


def beta_schedule(t, d=2, delta=0.1):
    return 2 * math.log((t ** (d / 2 + 2)) * (math.pi ** 2) / (3 * delta))

# -----------------------------
# Core BO loop
# -----------------------------

def run_bo(f, policy, T=100, lengthscale=1.0, n_init=5):
    train_x = sample_uniform(n_init, f.bounds).to("cpu")
    train_y = f(train_x).to("cpu")

    min_regret = []
    regret = []

    for t in range(1, T + 1):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=1e-6).to("cpu")
        model = ExactGPModel(train_x, train_y, likelihood, lengthscale).to("cpu")
        model.eval(); likelihood.eval()

        Xcand = sample_uniform(5000, f.bounds).to("cpu")

        if policy == 'ucb':
            beta = beta_schedule(t)
            acq = gp_ucb(model, likelihood, Xcand, beta)
            x_next = Xcand[torch.argmin(acq)]
        elif policy == 'ei':
            y_best = train_y.min()
            acq = expected_improvement(model, likelihood, Xcand, y_best)
            x_next = Xcand[torch.argmax(acq)]
        elif policy == 'random':
            x_next = sample_uniform(1, f.bounds).squeeze(0)
        else:
            raise ValueError

        y_next = f(x_next.unsqueeze(0))

        train_x = torch.cat([train_x, x_next.unsqueeze(0)], dim=0)
        train_y = torch.cat([train_y, y_next], dim=0)

        # simple_regret = train_y.min().item() - f.star
        min_regret.append(train_y.min().item() - f.star)
        regret.append(y_next.item() - f.star)

    return np.array(regret), np.array(min_regret)

# -----------------------------
# Batched BO
# -----------------------------

def run_bo_batch(n=10, **kwargs):
    min_regrets = []
    regrets = []

    for _ in tqdm(range(n), desc="Running BO batch"):
        regret, min_regret = run_bo(**kwargs)
        min_regrets.append(min_regret)
        regrets.append(regret)
    
    return np.stack(regrets), np.stack(min_regrets)
