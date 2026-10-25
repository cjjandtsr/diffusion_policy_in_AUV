import unittest

import torch

from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.value import TemporalValueNetwork
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.rollout_buffer import OnPolicyRolloutBuffer
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.critic_state import with_history_scalar
from diffusion_policy_3d.rl.dppo_uav_rebuild.engine.initialization_parity import (
    _rng_snapshot, compare_a0_update0_state, construct_r1_critic_with_parity,
)


class R1CriticTest(unittest.TestCase):
    def test_optional_existing_a0_update0_state_must_match(self):
        actor = torch.nn.Parameter(torch.ones(3))
        torch.manual_seed(19)
        critic, _ = construct_r1_critic_with_parity(
            32, "cpu", (("actor", actor),))
        bundle = type("Bundle", (), {
            "adapter_state_dict": lambda self: {"actor": actor.detach().clone()}
        })()
        state = {
            "lora": bundle.adapter_state_dict(),
            "critic": {name: value.clone() for name, value in critic.state_dict().items()
                       if name != "history_weight"},
            "rng_cpu": torch.get_rng_state().clone(),
            "rng_cuda": None,
        }
        result = compare_a0_update0_state(
            state, bundle, critic, _rng_snapshot(torch.device("cpu")))
        self.assertEqual(result["status"], "A0_R1_UPDATE0_EXACT")
        state["lora"]["actor"][0] = 2.
        with self.assertRaises(RuntimeError):
            compare_a0_update0_state(
                state, bundle, critic, _rng_snapshot(torch.device("cpu")))

    def test_build_gate_hashes_actor_critic_and_rng(self):
        actor = torch.nn.Parameter(torch.ones(3))
        torch.manual_seed(11)
        expected = TemporalValueNetwork(hidden_dim=32)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(11)
        actual, evidence = construct_r1_critic_with_parity(
            32, "cpu", (("actor", actor),))
        self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(value, actual.state_dict()[name]), name)
        self.assertEqual(evidence["a0_critic_parameter_sha256"],
                         evidence["r1_critic_base_parameter_sha256"])
        self.assertEqual(evidence["value_max_abs_delta"], 0.)
        self.assertEqual(len(evidence["actor_parameter_sha256"]), 64)
        self.assertEqual(len(evidence["rng_state_sha256"]), 64)
    def test_current_and_next_beta_are_distinct_and_actor_condition_is_untouched(self):
        actor = {"state": torch.zeros(1, 576), "global_cond": torch.zeros(1, 576)}
        before = with_history_scalar(actor, {"initial_distance": 10., "best_distance": 8.})
        after = with_history_scalar(actor, {"initial_distance": 10., "best_distance": 6.})
        self.assertEqual(set(actor), {"state", "global_cond"})
        self.assertAlmostEqual(float(before["beta"][0, 0]), .8)
        self.assertAlmostEqual(float(after["beta"][0, 0]), .6)

    def test_zero_history_branch_preserves_a0_value_and_rng(self):
        observation = torch.arange(576, dtype=torch.float32).reshape(1, 576) / 576.
        torch.manual_seed(77)
        old = TemporalValueNetwork(hidden_dim=32)
        old_rng = torch.get_rng_state().clone()
        torch.manual_seed(77)
        new = TemporalValueNetwork(hidden_dim=32, history_scalar=True)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        for key, value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[key]), key)
        for beta in (0., .2, 1.):
            inputs = {"global_cond": observation, "beta": torch.tensor([[beta]])}
            self.assertTrue(torch.equal(old({"global_cond": observation}), new(inputs)))

    def test_r1_buffer_keeps_critic_beta_with_actor_condition(self):
        buffer = OnPolicyRolloutBuffer(1, outer_update_id=1, history_scalar=True)
        buffer.add(
            condition={"state": torch.zeros(576), "global_cond": torch.zeros(576),
                       "beta": torch.tensor([.75])},
            chain=torch.zeros(11, 16, 11), old_logprobs=torch.zeros(10, 16, 11),
            reward=1., value=0., next_value=0., terminated=True, truncated=False)
        buffer.finalize()
        pairs = buffer.materialize_pairs([0], [0], device="cpu", pass_id=0,
                                         mode="sampled_k", microbatch_pairs=1,
                                         schedule_audit={})
        self.assertEqual(tuple(pairs.condition["beta"].shape), (1, 1))
        self.assertEqual(float(pairs.condition["beta"][0, 0]), .75)


if __name__ == "__main__":
    unittest.main()
