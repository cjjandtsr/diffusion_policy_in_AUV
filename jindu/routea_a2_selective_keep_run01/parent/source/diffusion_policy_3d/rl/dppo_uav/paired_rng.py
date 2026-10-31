"""Paired policy-noise RNG for paired evaluation (review R5.4 / §4.5).

BC and RL must replay the SAME diffusion noise sequence for the same
bank entry — otherwise the paired comparison mixes environment and
policy-noise variance. Derivation (review §7 R5.4):

    policy_noise_seed = uint64(SHA256(bank_id || entry_id || replica_id)[0:8])

Per-side isolation: each side saves the global CPU/CUDA/python/numpy RNG
states, reseeds them from the derived seed, runs its episode, then
restores the globals (the environment's own RNG is untouched — it is
already paired via the frozen episode config/seed replay).

RNG_CONTRACT_VERSION records the seeding contract so an artifact from a
different contract is never silently compared.
"""

import hashlib
import random
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch

RNG_CONTRACT_VERSION = "paired-rng-v1"


def policy_noise_seed(
    bank_id: str, entry_id: str, replica_id: int = 0,
) -> int:
    """uint64(SHA256(bank_id || entry_id || replica_id)[0:8])."""
    payload = f"{bank_id}||{entry_id}||{int(replica_id)}".encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "little"
    )


@contextmanager
def paired_rng_context(seed: int) -> Iterator[int]:
    """Run a policy episode under an isolated RNG context: every global
    stream is saved, reseeded from ``seed``, then restored — so BC and RL
    draw the IDENTICAL diffusion noise sequence for the same seed while
    the surrounding process keeps its own RNG stream untouched.

    Yields the seed; the caller records sampler-call counts itself.
    """
    torch_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    np_state = np.random.get_state()
    py_state = random.getstate()
    # numpy/python accept 32-bit seeds only — every stream consumes the
    # same 32-bit projection of the recorded 64-bit derived seed.
    seed32 = int(seed) & 0xFFFFFFFF
    torch.manual_seed(seed32)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed32)
    np.random.seed(seed32)
    random.seed(seed32)
    try:
        yield int(seed)
    finally:
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        random.setstate(py_state)
