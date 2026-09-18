"""
Rectified Flow from Scratch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""End-to-end demo: train a tiny rectified flow on a 2-D two-blob mixture and sample.

Gaussian noise is unstructured (high nearest-neighbor MSE to the blobs).
After a short SGD run the Euler ODE sampler lands much closer to the data.
Reflow then pairs that original noise with the generated endpoints.
"""
import torch


def main():
    torch.manual_seed(0)
    centers = torch.tensor([[-2.5, 0.0], [2.5, 0.0]], dtype=torch.float32)
    data = make_mixture_dataset(centers, n_per_component=16, std=0.4, seed=0)
    time_embed_dim = 8
    params = init_velocity_mlp(2, 32, time_embed_dim, seed=0)
    params, history = train_rectified_flow(
        params,
        data,
        n_steps=200,
        batch_size=16,
        lr=0.05,
        time_embed_dim=time_embed_dim,
        seed=1,
    )
    noise = sample_gaussian_noise(data[:16], seed=3)
    samples = euler_sample(noise, params, n_steps=20, time_embed_dim=time_embed_dim)
    noise_mse = float(sample_quality_mse(noise, data))
    sample_mse = float(sample_quality_mse(samples, data))
    print("steps:", len(history))
    print(f"loss: {history[0]:.4f} -> {history[-1]:.4f}")
    print(f"noise baseline MSE:  {noise_mse:.4f}")
    print(f"trained sample MSE:  {sample_mse:.4f}")
    print(f"improvement (noise - sample): {noise_mse - sample_mse:.4f}")
    x0_rf, x1_hat = make_reflow_pairs(noise, params, 20, time_embed_dim)
    print("reflow pair shapes:", tuple(x0_rf.shape), tuple(x1_hat.shape))
    result = rectified_flow_experiment(
        n_per_component=16,
        hidden_dim=32,
        time_embed_dim=8,
        n_train_steps=200,
        batch_size=16,
        lr=0.05,
        n_euler_steps=20,
        seed=0,
    )
    print("experiment sample/baseline:", result)


if __name__ == "__main__":
    main()

