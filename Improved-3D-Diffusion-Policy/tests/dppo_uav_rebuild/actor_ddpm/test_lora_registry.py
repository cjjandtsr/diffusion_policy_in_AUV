import torch
from torch import nn

from diffusion_policy_3d.rl.dppo_uav_rebuild.actor_ddpm.lora import install_lora


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 7)
        self.conv = nn.Conv1d(3, 4, 3, padding=1)


def test_zero_lora_is_exact_and_only_adapters_are_trainable():
    torch.manual_seed(4)
    model = _Toy().double().eval()
    x_linear = torch.randn(2, 5, dtype=torch.float64)
    x_conv = torch.randn(2, 3, 8, dtype=torch.float64)
    expected = (model.linear(x_linear), model.conv(x_conv))

    bundle = install_lora(
        model,
        rank=2,
        alpha=4.0,
        expected_linear=1,
        expected_conv1d=1,
    )
    with bundle.mode(True):
        actual = (model.linear(x_linear), model.conv(x_conv))

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.endswith(("lora_a", "lora_b")) for name in trainable)
    assert bundle.target_names == ("conv", "linear")
    assert all(value.dtype == torch.float64 for _, value in bundle.named_adapter_parameters())
