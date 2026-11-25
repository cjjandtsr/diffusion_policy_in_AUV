"""Registered F8 treatment and matched control settings."""
from .f7_support import KEEP, REFERENCE_SHA

RUNS = {'P20': (202611081, True), 'P21': (202611082, True),
        'P22': (202611081, False), 'P23': (202611082, False)}


def validate_f8(config):
    f = config.f8
    if (set(f) != {'run_id', 'bank_path', 'bank_sha256', 'new_seed',
                    'keep_coefficient', 'reference_parent_receipt_sha256'}
            or (f.run_id not in RUNS and 'f9' not in config)):
        raise ValueError('invalid F8 registration')
    seed, treatment = ((int(config.run.seed), False) if 'f9' in config else RUNS[f.run_id])
    if (int(config.run.seed) != seed or str(config.reward_version) != 'v2'
            or int(config.run.updates) != 320 or int(config.run.rollout_transitions) != 1024
            or float(config.optimizer.actor_lr) != 1e-5
            or float(config.optimizer.actor_lr_end) != 1e-5
            or float(config.optimizer.critic_lr) != 1e-3
            or float(f.keep_coefficient) != KEEP
            or config.a2.reference_sha256 != REFERENCE_SHA
            or int(config.a2.keep_seed) != 2026102301 or config.a2.arm != 'T'
            or float(config.task_protocol.space_cost_budget) != 1.
            or any(k in config for k in ('f5','f6','f7','xk','task_bank'))
            or bool(f.bank_path) != treatment or bool(f.bank_sha256) != treatment
            or set(config.dppo) & {'ppo_inert_mask','self_imitation','inert_anchor',
                                   'min_std_vector','denoised_clip_value','drift_monitor','drift_gate'}):
        raise ValueError('F8 differs from registered settings')
