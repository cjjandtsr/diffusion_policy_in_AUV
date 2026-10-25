"""R6-G (F-05): canonical 60/side eval 生产者（§11.8.2 / §11.9 G 行）。

R6-D 只建了 12×5 矩阵校验器（validate_paired_matrix，expected
replicas=5），但生产路径缺失：run_paired_eval 硬编码 replica_id=0，
任何 eval 命令都只能产出 12 局/侧 —— 12×5=60/side 的矩阵永远无法
被生产。本测试锁定 R6-G 补上的生产者（scripts/eval_dppo_uav_official
._run_replica_evals + run_paired_eval 的 replica_id 参数化）：

- replica_id 0..N-1 逐次传递（R5.4 policy-noise 种子按 replica 派生）；
- 记录合并为一个矩阵（不丢、不重）；
- eval_invalid 跨 replica 累计；
- replicas=1 与 pre-R6-G 单 replica 调用语义一致（向后兼容）。

契约层（CPU）：patch run_paired_eval 验证循环/传递/合并逻辑。
集成层（GPU 真 env 的 replica 语义）由 R6.4 canary 的 60/side 真 eval
承担 —— canary eval artifact 的 12×5 矩阵 + evaluator 的 R5.4 种子
重推导校验就是集成验证（tests-mirror-training：不重复造 mock 环境）。
"""

from unittest.mock import patch

from scripts.eval_dppo_uav_official import _run_replica_evals


class _FakeRecord:
    def __init__(self, entry_id, replica):
        self.entry_id = entry_id
        self.replica_id = replica


def _fake_run_paired_eval(eval_env, bc_policy, rl_policy, contract,
                          loop_config, bank, bank_id="", replica_id=0,
                          policy_checkpoint_hash=""):
    # 每 replica 为每个 entry 产出一条记录（replica 语义：同一 bank
    # entry 不同噪声流 → 记录可区分）
    return (
        [_FakeRecord(e, replica_id) for e in range(len(bank))],
        0,
    )


def test_replica_loop_invokes_all_replicas_with_ids():
    bank = [{"id": i} for i in range(2)]
    with patch("scripts.eval_dppo_uav_official.run_paired_eval",
               side_effect=_fake_run_paired_eval) as mocked:
        records, invalid = _run_replica_evals(
            None, None, None, None, None, bank, bank_id="b", replicas=5,
        )
    assert mocked.call_count == 5
    ids = [c.kwargs["replica_id"] for c in mocked.call_args_list]
    assert ids == [0, 1, 2, 3, 4]
    # 5 replicas x 2 entries = 10 records，12×5 矩阵语义
    assert len(records) == 10
    assert invalid == 0
    replica_ids = sorted({r.replica_id for r in records})
    assert replica_ids == [0, 1, 2, 3, 4]


def test_records_merge_no_loss_no_dup():
    bank = [{"id": i} for i in range(3)]
    with patch("scripts.eval_dppo_uav_official.run_paired_eval",
               side_effect=_fake_run_paired_eval):
        records, _ = _run_replica_evals(
            None, None, None, None, None, bank, bank_id="b", replicas=2,
        )
    assert len(records) == 6
    keys = sorted((r.entry_id, r.replica_id) for r in records)
    assert len(keys) == len(set(keys))  # 无重复 (entry, replica)


def test_invalid_count_accumulates_across_replicas():
    def fake_invalid(env, bc, rl, contract, loop_config, bank,
                     bank_id="", replica_id=0, policy_checkpoint_hash=""):
        return ([_FakeRecord(0, replica_id)], 1)

    with patch("scripts.eval_dppo_uav_official.run_paired_eval",
               side_effect=fake_invalid):
        _, invalid = _run_replica_evals(
            None, None, None, None, None, [{"id": 0}],
            bank_id="b", replicas=3,
        )
    assert invalid == 3


def test_replicas_one_backward_compatible():
    # replicas=1 → 恰好一次 replica_id=0 调用（pre-R6-G 行为 bitwise 不变）
    with patch("scripts.eval_dppo_uav_official.run_paired_eval",
               side_effect=_fake_run_paired_eval) as mocked:
        records, _ = _run_replica_evals(
            None, None, None, None, None, [{"id": 0}],
            bank_id="b", replicas=1,
        )
    assert mocked.call_count == 1
    assert mocked.call_args.kwargs["replica_id"] == 0
    assert len(records) == 1
