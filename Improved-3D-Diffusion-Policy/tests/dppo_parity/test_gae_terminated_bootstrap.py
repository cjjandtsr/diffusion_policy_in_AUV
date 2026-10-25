"""R6-E §11.6.4/§11.6.5 (F-10): GAE bootstrap 只用 terminated mask。

§11.6.5 第 6 条：统计分类不得改变 bootstrap oracle。compute_gae 的
bootstrap_mask = 1.0 - float(terminated[t])（gae.py:93）是 R5.8-B
[PROVEN] 语义：truncated（time limit / 外部截断）只终止 λ 链
（continuation），绝不掩蔽 V(s_{t+1}) 自举。

这三条手算回归把语义钉死（t0 的 delta 是 9.0 vs 9.9 的可测分界）：
  - terminated=True 掩蔽 bootstrap：10 + 0.9*0*1 - 1 = 9.0；
  - truncated=True 不掩蔽 bootstrap：10 + 0.9*1*1 - 1 = 9.9，
    但 continuation=False 链在此终止；
  - 两标志皆 False 时链正常延续：adv[0] = 0.8 + 0.855*4.8。
任何把 truncated 掺进 bootstrap 的改动都会立刻改变这些值。

Run: pytest -q tests/dppo_parity/test_gae_terminated_bootstrap.py
"""

import numpy as np

from diffusion_policy_3d.rl.dppo_uav.gae import compute_gae


def _inputs(terminated, truncated):
    return dict(
        rewards=np.array([10.0, 5.0]),
        values=np.array([1.0, 1.0]),
        next_values=np.array([1.0, 1.0]),
        terminated=np.array(terminated, dtype=bool),
        truncated=np.array(truncated, dtype=bool),
        episode_ids=["e", "e"],
        gamma=0.9,
        gae_lambda=0.95,
    )


def test_terminated_masks_bootstrap():
    """MDP terminal 掩蔽自举：t0 的 delta 不含 V(s1)（9.0 而非 9.9），
    且 continuation 为 False — 终局状态不向后传播价值。"""
    adv, ret = compute_gae(**_inputs([True, False], [False, False]))
    # t1: delta = 5 + 0.9*1*1 - 1 = 4.9（链尾，adv = delta）
    # t0: delta = 10 + 0.9*0*1 - 1 = 9.0（terminated 掩蔽 bootstrap）
    np.testing.assert_allclose(adv, [9.0, 4.9], rtol=0, atol=1e-12)
    np.testing.assert_allclose(ret, [10.0, 5.9], rtol=0, atol=1e-12)


def test_truncated_does_not_mask_bootstrap_but_stops_chain():
    """time limit 不掩蔽自举（t0 的 delta = 9.9，与 terminated 情形
    的 9.0 区分），但 continuation=False 终止 λ 链 — truncated 绝不
    等价于 MDP terminal。"""
    adv, _ = compute_gae(**_inputs([False, False], [True, False]))
    # t0: delta = 10 + 0.9*1*1 - 1 = 9.9（bootstrap 保留）
    #     continuation = same_episode ∧ ¬term ∧ ¬trunc = False
    np.testing.assert_allclose(adv, [9.9, 4.9], rtol=0, atol=1e-12)


def test_continuation_chains_when_neither_flag_set():
    """两标志皆 False 时链正常延续（同一 episode 内 λ 递归），
    adv[0] = 0.8 + 0.9*0.95*4.8 = 4.904 — 反向证明上两条的差异
    确实来自 terminated/truncated 两个掩码。"""
    adv, _ = compute_gae(
        rewards=np.array([0.0, 5.0]),
        values=np.array([1.0, 2.0]),
        next_values=np.array([2.0, 2.0]),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        episode_ids=["e", "e"],
        gamma=0.9,
        gae_lambda=0.95,
    )
    # t1: delta = 5 + 0.9*2 - 2 = 4.8；adv[1] = 4.8
    # t0: delta = 0 + 0.9*2 - 1 = 0.8；
    #     adv[0] = 0.8 + 0.855 * 4.8 = 4.904
    np.testing.assert_allclose(adv, [4.904, 4.8], rtol=0, atol=1e-12)
