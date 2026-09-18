"""Sanity tests covering every stage of the pipeline defined in model.py:
interpolant -> objective -> model -> training -> ODE samplers -> evaluation/reflow.
Run with: pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model import (
    interpolate_linear,
    target_velocity,
    make_flow_batch,
    flow_matching_loss,
    sinusoidal_time_embedding,
    init_velocity_mlp,
    velocity_mlp_forward,
    make_mixture_dataset,
    train_rectified_flow,
    euler_step,
    euler_sample,
    heun_step,
    heun_sample,
    sample_quality_mse,
    make_reflow_pairs,
    rectified_flow_experiment,
)


def test_interpolant_endpoints():
    x0 = torch.zeros(4, 2)
    x1 = torch.ones(4, 2) * 5
    assert torch.allclose(interpolate_linear(x0, x1, torch.zeros(4, 1)), x0)
    assert torch.allclose(interpolate_linear(x0, x1, torch.ones(4, 1)), x1)


def test_target_velocity_is_constant_difference():
    x0 = torch.tensor([[3.0, 1.0]])
    x1 = torch.tensor([[1.0, 4.0]])
    assert torch.allclose(target_velocity(x0, x1), torch.tensor([[-2.0, 3.0]]))


def test_flow_batch_shapes():
    x1 = torch.randn(8, 3)
    xt, v_star, t = make_flow_batch(x1, seed=0)
    assert xt.shape == (8, 3)
    assert v_star.shape == (8, 3)
    assert t.shape == (8, 1)


def test_loss_is_zero_when_predictions_match():
    v = torch.randn(5, 2)
    assert flow_matching_loss(v, v).item() == 0.0


def test_time_embedding_shape_and_range():
    t = torch.rand(6, 1)
    emb = sinusoidal_time_embedding(t, embed_dim=16)
    assert emb.shape == (6, 16)
    assert emb.abs().max() <= 1.0 + 1e-6


def test_mlp_forward_shape_and_grad_flow():
    params = init_velocity_mlp(in_dim=2, hidden_dim=32, time_embed_dim=8, seed=0)
    x = torch.randn(5, 2)
    t = torch.rand(5, 1)
    v = velocity_mlp_forward(x, t, params, time_embed_dim=8)
    assert v.shape == (5, 2)
    v.sum().backward()
    assert params["W1"].grad is not None


def test_training_reduces_loss_on_average():
    centers = torch.tensor([[-2.5, 0.0], [2.5, 0.0]])
    data = make_mixture_dataset(centers, n_per_component=32, std=0.4, seed=0)
    params = init_velocity_mlp(2, 32, 8, seed=1)
    params, history = train_rectified_flow(
        params, data, n_steps=300, batch_size=16, lr=0.05, time_embed_dim=8, seed=2
    )
    early = sum(history[:20]) / 20
    late = sum(history[-20:]) / 20
    assert late < early


def test_euler_sample_matches_manual_constant_velocity_field():
    x0 = torch.zeros(1, 2)
    params = {
        "W1": torch.zeros(6, 2), "b1": torch.zeros(2),
        "W2": torch.zeros(2, 2), "b2": torch.zeros(2),
        "W3": torch.zeros(2, 2), "b3": torch.tensor([1.0, -0.5]),
    }
    out = euler_sample(x0, params, n_steps=4, time_embed_dim=4)
    assert torch.allclose(out, torch.tensor([[1.0, -0.5]]), atol=1e-5)


def test_heun_matches_euler_on_zero_dt():
    x = torch.tensor([[5.0, -1.0]])
    params = init_velocity_mlp(2, 8, 4, seed=0)
    out = heun_step(x, torch.tensor(0.4), torch.tensor(0.0), params, time_embed_dim=4)
    assert torch.allclose(out, x)


def test_sample_quality_mse_matches_hand_computation():
    samples = torch.tensor([[0.0, 0.0]])
    data = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    assert sample_quality_mse(samples, data).item() == 1.0


def test_reflow_pairs_preserve_noise_and_shape():
    x0 = torch.zeros(6, 2)
    params = init_velocity_mlp(2, 8, 4, seed=0)
    x0_out, x1_hat = make_reflow_pairs(x0, params, n_steps=4, time_embed_dim=4)
    assert torch.equal(x0_out, x0)
    assert x1_hat.shape == (6, 2)


def test_capstone_experiment_beats_noise_baseline():
    sample_mse, baseline_mse = rectified_flow_experiment(
        n_per_component=16, hidden_dim=32, time_embed_dim=8,
        n_train_steps=200, batch_size=16, lr=0.05, n_euler_steps=20, seed=0,
    )
    assert isinstance(sample_mse, float)
    assert sample_mse < baseline_mse
