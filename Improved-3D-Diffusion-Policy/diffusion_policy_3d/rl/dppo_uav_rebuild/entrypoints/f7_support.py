"""Registered F7 settings and checkpointed next-update actor multiplier."""
KEEP = .01061006086489362
REFERENCE_SHA = '8759fc40c20a20c3bd08a5568965f5562c508f3577d8c82fa4c646d6a080ce95'
RUNS = {'P16': (202611081, True), 'P17': (202611082, True),
        'P18': (202611081, False), 'P19': (202611082, False)}


def collection_details(info, env, action, position_before, weight):
    import numpy as np
    from diffusion_policy_3d.env.reward_v5 import approach_cost, segment_durations
    dt = segment_durations(info, env.traj_dt)
    cost, c = approach_cost(info['executed_path'], env.obstacles, dt, weight)
    position = np.asarray(position_before, dtype=np.float32)
    planned = [position.copy()]
    for delta in action.policy[0].detach().cpu().numpy()[:, :3]:
        position = position + delta
        planned.append(position.copy())
    return dict(segment_durations=dt.tolist(), approach_cbf_diagnostic=cost,
                point_clearances=c.tolist(), planned_path=np.asarray(planned).tolist())


class F7Schedule:
    def __init__(self, adaptive):
        self.adaptive = bool(adaptive)
        self.used_multiplier = self.next_multiplier = 5. if adaptive else 1.
        self.clip_fraction = None

    def finish(self, clip_fraction):
        self.used_multiplier = self.next_multiplier
        self.clip_fraction = float(clip_fraction)
        if self.adaptive:
            if clip_fraction < .05: self.next_multiplier = min(20.,self.next_multiplier*1.5)
            elif clip_fraction > .15: self.next_multiplier = max(1.,self.next_multiplier/1.5)

    def state_dict(self):
        return dict(adaptive=self.adaptive, used_multiplier=self.used_multiplier,
                    next_multiplier=self.next_multiplier, clip_fraction=self.clip_fraction)

    def load_state_dict(self, state):
        if bool(state['adaptive']) != self.adaptive:
            raise ValueError('F7 LR rule changed on resume')
        self.used_multiplier = state['used_multiplier']
        self.next_multiplier = state['next_multiplier']
        self.clip_fraction = state['clip_fraction']


def validate_f7(config):
    if 'f7' not in config:
        if str(config.get('reward_version')) == 'v5':
            raise ValueError('v5 requires registered F7')
        return
    f = config.f7
    seed, adaptive = RUNS[f.run_id]
    if (set(f) != {'run_id','approach_weight','adaptive_lr','keep_coefficient',
                   'reference_parent_receipt_sha256'}
            or int(config.run.seed) != seed or bool(f.adaptive_lr) != adaptive
            or str(config.reward_version) != ('v5' if adaptive else 'v2')
            or float(f.approach_weight) not in (1.,2.) or float(f.keep_coefficient) != KEEP
            or int(config.run.updates) != 320 or int(config.run.rollout_transitions) != 1024
            or float(config.optimizer.actor_lr) != 1e-5 or float(config.optimizer.actor_lr_end) != 1e-5
            or float(config.optimizer.critic_lr) != 1e-3
            or config.a2.reference_sha256 != REFERENCE_SHA or int(config.a2.keep_seed) != 2026102301
            or config.a2.arm != 'T' or float(config.task_protocol.space_cost_budget) != 1.
            or any(k in config for k in ('f5','f6','xk','task_bank'))
            or set(config.dppo) & {'ppo_inert_mask','self_imitation','inert_anchor','min_std_vector',
                                  'denoised_clip_value','drift_monitor','drift_gate'}):
        raise ValueError('F7 differs from registered v1.2 settings')
