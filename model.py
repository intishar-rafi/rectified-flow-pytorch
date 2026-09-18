"""
Rectified Flow from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np
import torch

# Step 1 - interpolate_linear
def interpolate_linear(x0, x1, t):
    """Return the linear interpolant from noise at t=0 to data at t=1."""
    # TODO: Return the linear interpolant from noise at t=0 to data at t=1.
    return (1 - t) * x0 + t * x1

# Step 2 - target_velocity
def target_velocity(x0, x1):
    """Return the constant straight-path target velocity from noise x0 to data x1."""
    # TODO: Return the constant straight-path target velocity along the interpolant...
    return x1 - x0

# Step 3 - sample_uniform_time
def sample_uniform_time(batch_size, seed):
    """Draw a column of times uniformly from the unit interval."""
    # TODO: Draw a column of times uniformly from the unit interval...
    torch.manual_seed(seed)
    return torch.rand(batch_size, 1)

# Step 4 - sample_gaussian_noise
def sample_gaussian_noise(x, seed):
    """Return standard normal noise matching the shape of a given tensor using a given seed."""
    # TODO: Return standard normal noise matching a tensor using a given seed...
    torch.manual_seed(seed)
    return torch.randn_like(x)

# Step 5 - make_flow_batch
def make_flow_batch(x1, seed):
    """Given data x1, sample noise and time and return xt, target velocity, and t."""
    # TODO: Implement `make_flow_batch` to assemble one flow-matching training triple from a batch of data....
    batch = x1.shape[0]
    x0 = sample_gaussian_noise(x1, seed)
    t = sample_uniform_time(batch, seed)
    xt = interpolate_linear(x0, x1, t)
    v_star = target_velocity(x0, x1)
    return xt, v_star, t

# Step 6 - flow_matching_loss
def flow_matching_loss(v_pred, v_target):
    """Score a predicted velocity against the straight-path target."""
    # TODO: Implement `flow_matching_loss` to score predicted velocities against a target.
    return ((v_pred - v_target) ** 2).mean()

# Step 7 - sinusoidal_time_embedding
def sinusoidal_time_embedding(t, embed_dim):
    """Map each time in t to a sin-cos Fourier feature vector of length embed_dim."""
    # TODO: Map each time in t to a sin-cos Fourier feature vector of length embed_dim.
    t = t.reshape(-1, 1)
    half = embed_dim // 2
    i = torch.arange(half, dtype=t.dtype, device=t.device)
    freqs = torch.exp(-torch.log(torch.tensor(10000.0, dtype=t.dtype, device=t.device)) * i / half)
    angles = t * freqs
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

# Step 8 - init_velocity_mlp
def init_velocity_mlp(in_dim, hidden_dim, time_embed_dim, seed):
    """Return a seeded parameter dict for a two-hidden-layer velocity MLP."""
    # TODO: Return a seeded parameter dict for a two-hidden-layer velocity MLP....
    torch.manual_seed(seed)

    def glorot(n_in, n_out):
        w = torch.randn(n_in, n_out) * ((2.0 / (n_in + n_out)) ** 0.5)
        return w.to(torch.float32).requires_grad_(True)

    W1 = glorot(in_dim + time_embed_dim, hidden_dim)
    b1 = torch.zeros(hidden_dim, dtype=torch.float32, requires_grad=True)
    W2 = glorot(hidden_dim, hidden_dim)
    b2 = torch.zeros(hidden_dim, dtype=torch.float32, requires_grad=True)
    W3 = glorot(hidden_dim, in_dim)
    b3 = torch.zeros(in_dim, dtype=torch.float32, requires_grad=True)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}

# Step 9 - velocity_mlp_forward
def velocity_mlp_forward(x, t, params, time_embed_dim):
    """Return a velocity of the same shape as x from states, times, and MLP params."""
    # TODO: Embed t with the sinusoidal embedding, concatenate it with x...
    batch = x.shape[0]
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device)
    t = t.reshape(-1, 1).expand(batch, 1) if t.numel() == 1 else t.reshape(batch, 1)
    t_emb = sinusoidal_time_embedding(t, time_embed_dim)

    h = torch.cat([x, t_emb], dim=-1)
    h = h @ params["W1"] + params["b1"]
    h = torch.relu(h)
    h = h @ params["W2"] + params["b2"]
    h = torch.relu(h)
    v = h @ params["W3"] + params["b3"]
    return v

# Step 10 - make_mixture_dataset
def make_mixture_dataset(centers, n_per_component, std, seed):
    """Build a fixed-seed 2-D Gaussian mixture from component centers."""
    # TODO: Build a fixed-seed 2-D Gaussian mixture from component centers...
    torch.manual_seed(seed)
    n_components = centers.shape[0]
    rows = []
    for k in range(n_components):
        z = torch.randn(n_per_component, 2)
        rows.append(centers[k] + std * z)
    return torch.cat(rows, dim=0).to(torch.float32)

# Step 11 - flow_train_step
def flow_train_step(params, data, batch_size, lr, time_embed_dim, seed):
    """Draw a minibatch of data, compute the flow-matching loss through the velocity MLP, and take one manual SGD update."""
    # TODO: Implement flow_train_step to perform one in-place manual SGD update on a velocity MLP.
    torch.manual_seed(seed)
    n = data.shape[0]
    idx = torch.randint(0, n, (batch_size,))
    minibatch = data[idx]

    xt, v_star, t = make_flow_batch(minibatch, seed + 1)

    v_pred = velocity_mlp_forward(xt, t, params, time_embed_dim)
    loss = flow_matching_loss(v_pred, v_star)
    loss_value = loss.item()

    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    loss.backward()

    with torch.no_grad():
        for p in params.values():
            p -= lr * p.grad

    return params, loss_value

# Step 12 - train_rectified_flow
def train_rectified_flow(params, data, n_steps, batch_size, lr, time_embed_dim, seed):
    """Repeat flow_train_step for n_steps and return updated params plus the loss history."""
    # TODO: train_rectified_flow runs a short training loop for the velocity MLP...
    loss_history = []
    for step in range(n_steps):
        params, loss_value = flow_train_step(params, data, batch_size, lr, time_embed_dim, seed + step)
        loss_history.append(loss_value)
    return params, loss_history

# Step 13 - linspace_timesteps
def linspace_timesteps(n_steps):
    # TODO: Return n_steps plus one evenly spaced times from 0 to 1 inclusive.
    return torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float32)

# Step 14 - euler_step
def euler_step(x, v, dt):
    """Advance a state by one explicit Euler step."""
    # TODO: Advance a state by one explicit Euler step.
    return x + dt * v

# Step 15 - euler_sample
def euler_sample(x0, params, n_steps, time_embed_dim):
    """Integrate the velocity ODE from noise at t=0 to data at t=1 using evenly spaced Euler steps."""
    # TODO: Integrate the velocity ODE from noise at t=0 to data at t=1....
    times = linspace_timesteps(n_steps)
    x = x0
    batch = x0.shape[0]
    for i in range(n_steps):
        t = times[i].expand(batch, 1)
        dt = times[i + 1] - times[i]
        v = velocity_mlp_forward(x, t, params, time_embed_dim)
        x = euler_step(x, v, dt)
    return x

# Step 16 - heun_step
def heun_step(x, t, dt, params, time_embed_dim):
    """Take one Heun predictor-corrector step of the flow ODE.

    Args:
        x: (batch, dim) current state.
        t: scalar or 0-dim current time.
        dt: float or 0-dim step size.
        params: velocity-MLP parameter dict.
        time_embed_dim: even integer time-embedding size.

    Returns:
        (batch, dim) Heun-updated state.
    """
    # TODO: Take one Heun predictor-corrector step of the learned velocity ODE.
    batch = x.shape[0]
    t = torch.as_tensor(t, dtype=x.dtype, device=x.device).reshape(1, 1).expand(batch, 1)

    v1 = velocity_mlp_forward(x, t, params, time_embed_dim)
    x_pred = euler_step(x, v1, dt)

    t_next = t + dt
    v2 = velocity_mlp_forward(x_pred, t_next, params, time_embed_dim)

    return x + 0.5 * dt * (v1 + v2)

# Step 17 - heun_sample
def heun_sample(x0, params, n_steps, time_embed_dim):
    # TODO: Integrate the velocity ODE from noise at t=0 to data at t=1...
    times = linspace_timesteps(n_steps)
    x = x0
    for i in range(n_steps):
        t = times[i]
        dt = times[i + 1] - times[i]
        x = heun_step(x, t, dt, params, time_embed_dim)
    return x

# Step 18 - sample_quality_mse
def sample_quality_mse(samples, data):
    """Return the mean nearest-neighbor squared Euclidean distance from samples to data."""
    # TODO: Compute the mean nearest-neighbor squared Euclidean distance from samples to data.
    diffs = samples.unsqueeze(1) - data.unsqueeze(0)
    sq_dists = (diffs ** 2).sum(dim=-1)
    nn_dists = sq_dists.min(dim=1).values
    return nn_dists.mean()

# Step 19 - make_reflow_pairs
def make_reflow_pairs(x0, params, n_steps, time_embed_dim):
    """Build rectified (noise, generated-data) training pairs from noise."""
    # TODO: Build rectified (noise, generated-data) training pairs from noise.
    x1_hat = euler_sample(x0, params, n_steps, time_embed_dim)
    return x0, x1_hat

# Step 20 - rectified_flow_experiment
def rectified_flow_experiment(n_per_component, hidden_dim, time_embed_dim, n_train_steps, batch_size, lr, n_euler_steps, seed):
    """Train a 2-D flow on a two-blob mixture and return sample vs baseline MSE floats."""
    # TODO: Train a 2-D flow on a two-blob mixture and return sample vs baseline MSE floats.
    centers = torch.tensor([[-2.5, 0.0], [2.5, 0.0]], dtype=torch.float32)
    data = make_mixture_dataset(centers, n_per_component, 0.4, seed)

    in_dim = 2
    params = init_velocity_mlp(in_dim, hidden_dim, time_embed_dim, seed + 1)

    params, loss_history = train_rectified_flow(
        params, data, n_train_steps, batch_size, lr, time_embed_dim, seed + 2
    )

    noise = sample_gaussian_noise(data, seed + 3)

    samples = euler_sample(noise, params, n_euler_steps, time_embed_dim)

    sample_mse = sample_quality_mse(samples, data)
    baseline_mse = sample_quality_mse(noise, data)

    return float(sample_mse), float(baseline_mse)

