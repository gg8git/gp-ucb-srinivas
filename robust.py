from bo_utils import *
from functions import Branin, Ackley, Levy, Himmelblau
from logging_utils import save_regret, plot_regret

# -----------------------------
# Misspecified Lengthscale Experiment
# -----------------------------

T = 500
N = 20
f = Branin()

reg_ls_correct, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=1.0)
save_regret(reg_ls_correct, filename=f"log_{f.name.lower()}_robust_normal.npy")
reg_ls_short, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=0.1)
save_regret(reg_ls_short, filename=f"log_{f.name.lower()}_robust_short.npy")
reg_ls_long, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=10.0)
save_regret(reg_ls_long, filename=f"log_{f.name.lower()}_robust_long.npy")

# -----------------------------
# Plot Experiment
# -----------------------------

plot_regret(
    curves=[reg_ls_correct, reg_ls_short, reg_ls_long],
    labels=["Lengthscale = 1.0", "Lengthscale = 0.1", "Lengthscale = 10.0"],
    title=f"Experiment 2: BO Robustness to Lengthscale Misspecification on {f.name} (GP-UCB)",
    filename=f"experiment_{f.name.lower()}_robust.png",
    log_regret=True,
)

plot_regret(
    curves=[reg_ls_correct, reg_ls_short, reg_ls_long],
    labels=["Lengthscale = 1.0", "Lengthscale = 0.1", "Lengthscale = 10.0"],
    title=f"Experiment 2: BO Robustness to Lengthscale Misspecification on {f.name} (GP-UCB)",
    filename=f"experiment_{f.name.lower()}_cum_robust.png",
    cumulative=True,
)

# -----------------------------
# Different Functions
# -----------------------------

FUNCTIONS = [Ackley(), Levy(), Himmelblau()]

for f in FUNCTIONS:
    reg_ls_correct, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=1.0)
    save_regret(reg_ls_correct, filename=f"log_{f.name.lower()}_robust_normal.npy")
    reg_ls_short, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=0.1)
    save_regret(reg_ls_short, filename=f"log_{f.name.lower()}_robust_short.npy")
    reg_ls_long, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=10.0)
    save_regret(reg_ls_long, filename=f"log_{f.name.lower()}_robust_long.npy")

    plot_regret(
        curves=[reg_ls_correct, reg_ls_short, reg_ls_long],
        labels=["Lengthscale = 1.0", "Lengthscale = 0.1", "Lengthscale = 10.0"],
        title=f"Experiment 2: Lengthscale Misspecification Robustness on {f.name} (GP-UCB)",
        filename=f"experiment_{f.name.lower()}_robust.png",
        log_regret=True,
    )

    plot_regret(
        curves=[reg_ls_correct, reg_ls_short, reg_ls_long],
        labels=["Lengthscale = 1.0", "Lengthscale = 0.1", "Lengthscale = 10.0"],
        title=f"Experiment 2: Lengthscale Misspecification Robustness on {f.name} (GP-UCB)",
        filename=f"experiment_{f.name.lower()}_cum_robust.png",
        cumulative=True,
    )
