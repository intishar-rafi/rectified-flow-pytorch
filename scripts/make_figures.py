"""
Generates every plot used in the README, straight from model.py:
  assets/loss_curve.png          training loss vs. step
  assets/trajectories.png        noise -> data ODE trajectories (Euler)
  assets/before_after.png        noise vs. generated samples vs. real data
  assets/euler_vs_heun.png       sample-quality MSE vs. number of ODE steps

Run from the repo root:  python scripts/make_figures.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from model import (
    make_mixture_dataset,
    init_velocity_mlp,
    train_rectified_flow,
    sample_gaussian_noise,
    euler_sample,
    heun_sample,
    linspace_timesteps,
    velocity_mlp_forward,
    euler_step,
    sample_quality_mse,
)

warnings.filterwarnings("ignore")

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
os.makedirs(ASSETS, exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#0d1117",
        "savefig.facecolor": "#0d1117",
        "axes.edgecolor": "#484f58",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.size": 12,
        "axes.grid": True,
        "grid.linewidth": 0.6,
    }
)

SEED = 0
N_PER_COMPONENT = 48
HIDDEN_DIM = 64
TIME_EMBED_DIM = 16
N_TRAIN_STEPS = 800
BATCH_SIZE = 32
LR = 0.05
N_EULER_STEPS = 30

# ---------------------------------------------------------------- data + training
centers = torch.tensor([[-2.5, 0.0], [2.5, 0.0]], dtype=torch.float32)
data = make_mixture_dataset(centers, N_PER_COMPONENT, 0.4, SEED)

params = init_velocity_mlp(2, HIDDEN_DIM, TIME_EMBED_DIM, SEED + 1)
params, loss_history = train_rectified_flow(
    params, data, N_TRAIN_STEPS, BATCH_SIZE, LR, TIME_EMBED_DIM, SEED + 2
)

noise = sample_gaussian_noise(data, SEED + 3)
with torch.no_grad():
    samples = euler_sample(noise, params, N_EULER_STEPS, TIME_EMBED_DIM)
    sample_mse = float(sample_quality_mse(samples, data))
    baseline_mse = float(sample_quality_mse(noise, data))

print(f"loss: {loss_history[0]:.4f} -> {loss_history[-1]:.4f}")
print(f"noise baseline MSE: {baseline_mse:.4f}")
print(f"trained sample MSE: {sample_mse:.4f}")

# ---------------------------------------------------------------- 1. loss curve
window = 20
smoothed = [
    sum(loss_history[max(0, i - window):i + 1]) / len(loss_history[max(0, i - window):i + 1])
    for i in range(len(loss_history))
]

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(loss_history, color="#30363d", linewidth=1, label="raw")
ax.plot(smoothed, color="#58a6ff", linewidth=2.2, label=f"{window}-step moving avg")
ax.set_xlabel("training step")
ax.set_ylabel("flow-matching MSE loss")
ax.set_title("Velocity field training loss", fontsize=13, color="#e6edf3")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "loss_curve.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 2. ODE trajectories
torch.manual_seed(7)
n_traj = 24
x0_traj = torch.randn(n_traj, 2)
n_vis_steps = 40
times = linspace_timesteps(n_vis_steps)
path = torch.zeros(n_vis_steps + 1, n_traj, 2)
x = x0_traj
path[0] = x
with torch.no_grad():
    for i in range(n_vis_steps):
        t = times[i].expand(n_traj, 1)
        dt = times[i + 1] - times[i]
        v = velocity_mlp_forward(x, t, params, TIME_EMBED_DIM)
        x = euler_step(x, v, dt)
        path[i + 1] = x

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(data[:, 0], data[:, 1], s=14, color="#3fb950", alpha=0.55, label="real data", zorder=2)
for j in range(n_traj):
    ax.plot(path[:, j, 0], path[:, j, 1], color="#58a6ff", alpha=0.35, linewidth=1, zorder=1)
ax.scatter(path[0, :, 0], path[0, :, 1], s=22, color="#f85149", label="noise (t=0)", zorder=3)
ax.scatter(path[-1, :, 0], path[-1, :, 1], s=22, color="#d29922", label="generated (t=1)", zorder=3)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Learned ODE trajectories: noise → data", fontsize=13, color="#e6edf3")
ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "trajectories.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 3. before / after
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
axes[0].scatter(data[:, 0], data[:, 1], s=16, color="#3fb950", alpha=0.5, label="real data")
axes[0].scatter(noise[:, 0], noise[:, 1], s=16, color="#f85149", alpha=0.7, label="Gaussian noise")
axes[0].set_title("Before: pure noise", fontsize=12, color="#e6edf3")
axes[0].legend(frameon=False)

axes[1].scatter(data[:, 0], data[:, 1], s=16, color="#3fb950", alpha=0.5, label="real data")
axes[1].scatter(samples[:, 0], samples[:, 1], s=16, color="#d29922", alpha=0.8, label="generated samples")
axes[1].set_title(f"After {N_EULER_STEPS} Euler steps", fontsize=12, color="#e6edf3")
axes[1].legend(frameon=False)

for ax in axes:
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.suptitle("Nearest-neighbor MSE to real data: "
             f"noise {baseline_mse:.2f} → samples {sample_mse:.2f}",
             fontsize=12, color="#8b949e")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "before_after.png"), dpi=170)
plt.close(fig)

# ---------------------------------------------------------------- 4. Euler vs Heun, MSE vs steps
step_counts = [1, 2, 4, 6, 10, 16, 24, 40]
euler_mses, heun_mses = [], []
with torch.no_grad():
    for n in step_counts:
        s_euler = euler_sample(noise, params, n, TIME_EMBED_DIM)
        s_heun = heun_sample(noise, params, n, TIME_EMBED_DIM)
        euler_mses.append(float(sample_quality_mse(s_euler, data)))
        heun_mses.append(float(sample_quality_mse(s_heun, data)))

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(step_counts, euler_mses, "o-", color="#58a6ff", label="Euler (1st order)")
ax.plot(step_counts, heun_mses, "o-", color="#d29922", label="Heun (2nd order)")
ax.axhline(baseline_mse, color="#f85149", linestyle="--", linewidth=1.2, label="noise baseline")
ax.set_xlabel("number of ODE integration steps")
ax.set_ylabel("nearest-neighbor MSE to real data")
ax.set_title("Sample quality vs. integration budget", fontsize=13, color="#e6edf3")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "euler_vs_heun.png"), dpi=170)
plt.close(fig)

print("figures written to", ASSETS)
