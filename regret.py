from bo_utils import *
from functions import Branin, Ackley, Levy, Himmelblau
from logging_utils import save_regret, plot_regret

# -----------------------------
# Sublinear Regret Experiment
# -----------------------------

T = 500
N = 20
f = Branin()

reg_ucb, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=1.0)
save_regret(reg_ucb, filename=f"log_{f.name.lower()}_regret_ucb.npy")
reg_ei, _ = run_bo_batch(N, f=f, policy='ei', T=T, lengthscale=1.0)
save_regret(reg_ucb, filename=f"log_{f.name.lower()}_regret_ei.npy")
reg_rand, _ = run_bo_batch(N, f=f, policy='random', T=T, lengthscale=1.0)
save_regret(reg_ucb, filename=f"log_{f.name.lower()}_regret_rand.npy")

# -----------------------------
# Plot Experiment
# -----------------------------

plot_regret(
    curves=[reg_ucb, reg_ei, reg_rand],
    labels=["GP-UCB", "Expected Improvement", "Random Search"],
    title=f"Experiment 1: Sublinear Regret of BO on {f.name}",
    filename=f"experiment_{f.name.lower()}_regret.png",
    log_regret=True,
)

plot_regret(
    curves=[reg_ucb, reg_ei, reg_rand],
    labels=["GP-UCB", "Expected Improvement", "Random Search"],
    title=f"Experiment 1: Sublinear Regret of BO on {f.name}",
    filename=f"experiment_{f.name.lower()}_cum_regret.png",
    cumulative=True,
)

# -----------------------------
# Different Functions
# -----------------------------

FUNCTIONS = [Ackley(), Levy(), Himmelblau()]

for f in FUNCTIONS:
    reg_ucb, _ = run_bo_batch(N, f=f, policy='ucb', T=T, lengthscale=1.0)
    save_regret(reg_ucb, filename=f"log_{f.name.lower()}_regret_ucb.npy")
    reg_ei, _ = run_bo_batch(N, f=f, policy='ei', T=T, lengthscale=1.0)
    save_regret(reg_ei, filename=f"log_{f.name.lower()}_regret_ei.npy")
    reg_rand, _ = run_bo_batch(N, f=f, policy='random', T=T, lengthscale=1.0)
    save_regret(reg_rand, filename=f"log_{f.name.lower()}_regret_rand.npy")

    plot_regret(
        curves=[reg_ucb, reg_ei, reg_rand],
        labels=["GP-UCB", "Expected Improvement", "Random Search"],
        title=f"Experiment 1: Sublinear Regret of BO on {f.name}",
        filename=f"experiment_{f.name.lower()}_regret.png",
        log_regret=True,
    )

    plot_regret(
        curves=[reg_ucb, reg_ei, reg_rand],
        labels=["GP-UCB", "Expected Improvement", "Random Search"],
        title=f"Experiment 1: Sublinear Regret of BO on {f.name}",
        filename=f"experiment_{f.name.lower()}_cum_regret.png",
        cumulative=True,
    )
    