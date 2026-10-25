import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.value import TemporalValueNetwork


def test_value_network_is_not_permutation_invariant():
    torch.manual_seed(9)
    network = TemporalValueNetwork(history=3, frame_dim=192, hidden_dim=64)
    frames = torch.randn(4, 3, 192)

    forward = network({"global_cond": frames.reshape(4, 576)})
    reversed_time = network({"global_cond": frames.flip(1).reshape(4, 576)})

    assert forward.shape == (4, 1)
    assert not torch.equal(forward, reversed_time)
