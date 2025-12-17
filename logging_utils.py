# -----------------------------
# Plotting utilities
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12

def save_regret(regrets, filename="regrets.npy"):
    np.save(filename, regrets)
    print(f"Saved regrets to {filename}")

def plot_regret(curves, labels, title, filename, log_regret=False, cumulative=False):
    plt.figure(figsize=(9, 6))

    for r, label in zip(curves, labels):
        if cumulative:
            r = np.cumsum(r, axis=1)

        mean_curve = np.mean(r, axis=0)
        std_curve = np.std(r, axis=0)

        if log_regret:
            # Add eps to avoid log(0)
            log_r = np.log(r + EPS)  # shape (n, T)
            mean_curve = np.mean(log_r, axis=0)
            std_curve = np.std(log_r, axis=0)
        
        plt.plot(mean_curve, label=label)
        plt.fill_between(
            np.arange(len(mean_curve)),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )
    
    plt.xlabel("# of Oracle Calls")
    if log_regret:
        ylabel = ("Cumulative " if cumulative else "Simple ") + "Log Regret"
    else:
        ylabel = ("Cumulative " if cumulative else "Simple ") + "Regret"
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_cum_regret(curves, labels, title, filename, log_regret=False):
    plt.figure(figsize=(6, 4))

    for r, label in zip(curves, labels):
        mean_curve = np.mean(r, axis=0)
        std_curve = np.std(r, axis=0)

        if log_regret:
            # Add eps to avoid log(0)
            log_curves = np.log(r + EPS)  # shape (n, T)
            mean_curve = np.mean(log_curves, axis=0)
            std_curve = np.std(log_curves, axis=0)
        
        plt.plot(mean_curve, label=label)
        plt.fill_between(
            np.arange(len(mean_curve)),
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2
        )
    
    plt.xlabel("# of Oracle Calls")
    if log_regret:
        plt.ylabel("Simple Log Regret")
    else:
        plt.ylabel("Simple Regret")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()