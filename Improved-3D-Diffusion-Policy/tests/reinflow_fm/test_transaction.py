from __future__ import annotations

import random
import unittest

import numpy as np
import torch
from torch import nn

from diffusion_policy_3d.rl.reinflow_fm.audit import training_state_sha256
from diffusion_policy_3d.rl.reinflow_fm.transaction import UpdateTransaction


class UpdateTransactionTests(unittest.TestCase):
    def _system(self):
        actor = nn.Linear(3, 3)
        noise = nn.Linear(3, 3)
        critic = nn.Linear(3, 1)
        actor_optimizer = torch.optim.AdamW(
            list(actor.parameters()) + list(noise.parameters()), lr=1e-3
        )
        critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-3)
        # Materialize optimizer moments before capture.
        (actor(torch.ones(2, 3)).sum() + noise(torch.ones(2, 3)).sum()).backward()
        actor_optimizer.step()
        critic(torch.ones(2, 3)).sum().backward()
        critic_optimizer.step()
        actor_optimizer.zero_grad(set_to_none=True)
        critic_optimizer.zero_grad(set_to_none=True)
        counters = {"outer_update": 7, "env_steps": 128}
        return actor, noise, critic, (actor_optimizer, critic_optimizer), counters

    def test_nan_failure_restores_modules_optimizers_counters_and_rng(self):
        random.seed(3)
        np.random.seed(5)
        torch.manual_seed(7)
        actor, noise, critic, optimizers, counters = self._system()
        before = training_state_sha256(
            actor=actor,
            noise=noise,
            critic=critic,
            optimizers=optimizers,
            counters=counters,
            include_rng=True,
        )
        transaction = UpdateTransaction(
            actor=actor,
            noise=noise,
            critic=critic,
            optimizers=optimizers,
            counters=counters,
        ).capture()

        with torch.no_grad():
            next(actor.parameters()).fill_(float("nan"))
            next(noise.parameters()).add_(4.0)
            next(critic.parameters()).mul_(0.0)
        optimizers[0].param_groups[0]["lr"] = 9.0
        counters["outer_update"] = 999
        counters["new_counter"] = 1
        random.random()
        np.random.rand()
        torch.rand(4)

        transaction.rollback()
        after = training_state_sha256(
            actor=actor,
            noise=noise,
            critic=critic,
            optimizers=optimizers,
            counters=counters,
            include_rng=True,
        )
        self.assertEqual(after, before)

    def test_uncommitted_context_rolls_back_but_commit_keeps_update(self):
        actor, noise, critic, optimizers, counters = self._system()
        original = next(actor.parameters()).detach().clone()
        with UpdateTransaction(
            actor=actor,
            noise=noise,
            critic=critic,
            optimizers=optimizers,
            counters=counters,
        ):
            with torch.no_grad():
                next(actor.parameters()).add_(1)
        self.assertTrue(torch.equal(next(actor.parameters()), original))

        with UpdateTransaction(
            actor=actor,
            noise=noise,
            critic=critic,
            optimizers=optimizers,
            counters=counters,
        ) as transaction:
            with torch.no_grad():
                next(actor.parameters()).add_(1)
            transaction.commit()
        self.assertFalse(torch.equal(next(actor.parameters()), original))


if __name__ == "__main__":
    unittest.main()
