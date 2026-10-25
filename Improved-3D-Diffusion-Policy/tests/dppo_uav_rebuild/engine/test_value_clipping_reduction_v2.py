"""PPO value clipping must take per-pair max before the global mean."""

import torch


def test_max_before_mean_differs_from_wrong_mean_then_max():
    returns = torch.tensor([0.0, 10.0, 0.0, 10.0])
    old = torch.tensor([0.0, 0.0, 10.0, 10.0])
    new = torch.tensor([5.0, 5.0, 5.0, 5.0])
    clip = 0.2
    clipped = old + torch.clamp(new - old, -clip, clip)
    unclipped_loss = (new - returns).square()
    clipped_loss = (clipped - returns).square()
    correct = torch.maximum(unclipped_loss, clipped_loss).mean()
    wrong = torch.maximum(unclipped_loss.mean(), clipped_loss.mean())
    assert float((clipped_loss > unclipped_loss).float().mean()) >= 0.30
    assert correct != wrong
