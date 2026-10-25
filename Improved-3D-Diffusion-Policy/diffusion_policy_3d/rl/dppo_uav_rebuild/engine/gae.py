"""GAE with separate terminated and time-limit bootstrap semantics."""

import torch


def compute_gae(
    rewards, values, next_values, terminated, truncated, gamma: float, lam: float,
    bootstrap_allowed=None,
):
    tensors = (rewards, values, next_values, terminated, truncated)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("GAE inputs must be rank-1")
    if len({len(tensor) for tensor in tensors}) != 1:
        raise ValueError("GAE input lengths differ")
    if bootstrap_allowed is None:
        bootstrap_allowed = ~terminated.bool()
    if bootstrap_allowed.shape != rewards.shape or bootstrap_allowed.dtype != torch.bool:
        raise ValueError("bootstrap_allowed must be a matching boolean vector")
    if torch.any(bootstrap_allowed & terminated.bool()):
        raise ValueError("terminated transitions cannot bootstrap")
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
    for index in range(len(rewards) - 1, -1, -1):
        bootstrap = bootstrap_allowed[index].to(rewards.dtype)
        continuation = 1.0 - torch.logical_or(
            terminated[index], truncated[index]
        ).to(rewards.dtype)
        delta = rewards[index] + gamma * next_values[index] * bootstrap - values[index]
        carry = delta + gamma * lam * continuation * carry
        advantages[index] = carry
    return advantages, advantages + values
