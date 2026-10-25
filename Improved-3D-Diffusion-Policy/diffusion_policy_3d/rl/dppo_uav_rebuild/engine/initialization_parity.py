"""Construct an R1 critic with an executable A0 update-0 parity check."""

import hashlib

import torch

from .value import TemporalValueNetwork


def _tensor_hash(named_tensors):
    digest = hashlib.sha256()
    for name, value in named_tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _rng_snapshot(device):
    cpu = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    return cpu, cuda


def _restore_rng(snapshot, device):
    torch.set_rng_state(snapshot[0])
    if device.type == "cuda":
        torch.cuda.set_rng_state(snapshot[1], device)


def _rng_hash(snapshot):
    values = [("cpu", snapshot[0])]
    if snapshot[1] is not None:
        values.append(("cuda", snapshot[1]))
    return _tensor_hash(values)


def construct_r1_critic_with_parity(hidden_dim, device, actor_parameters, *, episode_context=False):
    """Build A0 reference and R1 on the same random stream; keep R1 state."""
    device = torch.device(device)
    actor_parameters = tuple(actor_parameters)
    actor_before = _tensor_hash(actor_parameters)
    initial_rng = _rng_snapshot(device)
    reference = TemporalValueNetwork(hidden_dim=hidden_dim).to(device)
    reference_rng = _rng_snapshot(device)
    reference_state = dict(reference.named_parameters())
    reference_hash = _tensor_hash(reference_state.items())

    _restore_rng(initial_rng, device)
    r1 = TemporalValueNetwork(hidden_dim=hidden_dim, history_scalar=not episode_context,
                             episode_context=episode_context).to(device)
    r1_rng = _rng_snapshot(device)
    r1_state = dict(r1.named_parameters())
    context_name = "context_weight" if episode_context else "history_weight"
    if set(r1_state) != set(reference_state) | {context_name}:
        raise RuntimeError("R1 critic changed A0 parameter schema")
    if any(not torch.equal(r1_state[name], value) for name, value in reference_state.items()):
        raise RuntimeError("R1 critic A0 parameter parity failed")
    if not torch.count_nonzero(r1_state[context_name]).item() == 0:
        raise RuntimeError("R1 history branch must start at zero")
    if _rng_hash(r1_rng) != _rng_hash(reference_rng):
        raise RuntimeError("R1 critic initialization consumed extra RNG")
    actor_after = _tensor_hash(actor_parameters)
    if actor_after != actor_before:
        raise RuntimeError("R1 critic construction changed actor parameters")
    condition = {
        "global_cond": torch.stack((
            torch.zeros(576, device=device),
            torch.linspace(-1., 1., 576, device=device),
            torch.ones(576, device=device))),
        "beta": torch.tensor([[0.], [.5], [1.]], device=device),
        "episode_context": torch.arange(18, dtype=torch.float32, device=device).reshape(3, 6) / 10.,
    }
    with torch.no_grad():
        expected = reference(condition)
        actual = r1(condition)
    tolerance = 0. if device.type == "cpu" else 1e-6
    if not torch.allclose(expected, actual, rtol=0., atol=tolerance):
        raise RuntimeError("R1/A0 update-0 value parity failed")
    evidence = {
        "actor_parameter_sha256": actor_after,
        "a0_critic_parameter_sha256": reference_hash,
        "r1_critic_base_parameter_sha256": _tensor_hash(
            (name, r1_state[name]) for name in reference_state),
        "rng_state_sha256": _rng_hash(r1_rng),
        "value_max_abs_delta": float((expected - actual).abs().max().item()),
    }
    return r1, evidence


def compare_a0_update0_state(state, lora_bundle, r1_critic, rng_snapshot):
    """Read-only comparison against an existing A0 update-0 checkpoint state."""
    a0_actor = state["lora"]
    r1_actor = lora_bundle.adapter_state_dict()
    a0_critic = state["critic"]
    r1_base = {name: value for name, value in r1_critic.state_dict().items()
               if name not in ("history_weight", "context_weight")}
    if set(a0_actor) != set(r1_actor) or set(a0_critic) != set(r1_base):
        raise RuntimeError("A0/R1 update-0 actor or critic parameter schema mismatch")
    actor_a0_hash = _tensor_hash(sorted(a0_actor.items()))
    actor_r1_hash = _tensor_hash(sorted(r1_actor.items()))
    critic_a0_hash = _tensor_hash(sorted(a0_critic.items()))
    critic_r1_hash = _tensor_hash(sorted(r1_base.items()))
    if actor_a0_hash != actor_r1_hash or critic_a0_hash != critic_r1_hash:
        raise RuntimeError("A0/R1 update-0 actor or critic bytes differ")
    rng_a0 = (state["rng_cpu"],
              state["rng_cuda"][torch.device(r1_critic.position.device).index or 0]
              if state["rng_cuda"] is not None else None)
    if _rng_hash(rng_a0) != _rng_hash(rng_snapshot):
        raise RuntimeError("A0/R1 update-0 Torch RNG state differs")
    return {
        "actor_parameter_sha256": actor_r1_hash,
        "critic_base_parameter_sha256": critic_r1_hash,
        "torch_rng_state_sha256": _rng_hash(rng_snapshot),
        "status": "A0_R1_UPDATE0_EXACT",
    }
