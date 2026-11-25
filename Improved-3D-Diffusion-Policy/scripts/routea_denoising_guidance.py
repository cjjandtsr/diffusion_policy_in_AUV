"""Evaluation-only x0 guidance injected into the restored native DDIM sampler."""
import hashlib
import inspect
import math
from pathlib import Path
import textwrap
import types

import numpy as np
import torch


def action_positions(full, unnormalize, position):
    """Differentiable action_to_world POS branch, preserving sequential FP32 adds."""
    delta = unnormalize(full)[..., 2:16, :3]
    current = torch.as_tensor(position, dtype=full.dtype, device=full.device)
    points = []
    for k in range(14):
        current = current + delta[..., k, :]
        points.append(current)
    return torch.stack(points, dim=-2)


def body_to_world(cloud, position, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = cloud.new_tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return cloud @ rotation.T + cloud.new_tensor(position)


def costs(points, cloud, previous=None, d_safe=.6):
    distance = torch.cdist(points, cloud).amin(dim=-1) if len(cloud) else None
    collision = (torch.relu(d_safe - distance).square().sum() if distance is not None
                 else points.sum() * 0.)
    collision = collision + torch.relu(.5 - points[..., 2]).square().sum()
    consistency = (points.sum() * 0. if previous is None else
                   (points[..., :10, :] - previous[..., 4:14, :]).square().sum(-1).mean())
    return collision, consistency


class Guidance:
    def __init__(self, unnormalize, eta, consistency, d_safe=.6, updates=2):
        self.unnormalize = unnormalize
        self.eta = eta
        self.consistency = consistency
        self.d_safe, self.updates = float(d_safe), int(updates)
        self.previous = None
        self.triggered_steps = []
        self.update_count = 0

    def begin(self, cloud, position, yaw):
        self.position = np.asarray(position, np.float32).copy()
        cloud = body_to_world(cloud, position, yaw)
        self.cloud = cloud[cloud[:, 2] >= .1].detach()
        self.triggered_steps = []
        self.update_count = 0

    def update(self, x0, index):
        i = int(index.reshape(-1)[0])
        if i < 7:
            return x0
        with torch.enable_grad():
            value = x0.detach().requires_grad_(True)
            p = action_positions(value, self.unnormalize, self.position)
            col, con = costs(p, self.cloud, self.previous, self.d_safe)
            cost = col + self.consistency * con
            if float(cost.detach()) == 0.:
                return x0
            self.triggered_steps.append(i)
            for update_index in range(self.updates):
                gradient, = torch.autograd.grad(cost, value)
                value = (value - self.eta * gradient / gradient.abs().amax().clamp_min(1e-6)).clamp(-2., 2.)
                value = value.detach().requires_grad_(True)
                self.update_count += 1
                if update_index + 1 < self.updates:
                    col, con = costs(action_positions(value, self.unnormalize, self.position),
                                     self.cloud, self.previous, self.d_safe)
                    cost = col + self.consistency * con
        return value.detach()


def install(runtime, guide):
    """Insert after native x0 clipping/optional epsilon handling; preserve all other operators."""
    original = runtime.p_mean_var
    assert runtime.use_ddim and not runtime.predict_epsilon and runtime.ddim_steps == 10
    assert not runtime.recalculate_epsilon_after_clip
    assert runtime.denoised_clip_value == 2.
    source = textwrap.dedent(inspect.getsource(original.__func__))
    anchor = '    # Clip epsilon for numerical stability'
    assert source.count(anchor) == 1
    injected = source.replace(anchor, '    x_recon = _probe_guidance.update(x_recon, index)\n\n' + anchor)
    namespace = dict(original.__func__.__globals__, _probe_guidance=guide)
    exec(compile(injected, '<evaluation-x0-guidance>', 'exec'), namespace)
    runtime.p_mean_var = types.MethodType(namespace['p_mean_var'], runtime)
    file = Path(inspect.getfile(original.__func__)).resolve()
    return dict(type=type(runtime).__qualname__, module=type(runtime).__module__, file=str(file),
                source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
                original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),
                injected_function_sha256=hashlib.sha256(injected.encode()).hexdigest(),
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                recalculate_epsilon_after_clip=runtime.recalculate_epsilon_after_clip,
                eps_clip_value=runtime.eps_clip_value)


class ProbePolicy:
    """Keep the formal policy's encode/seed/sample/unnormalize path intact."""
    def __init__(self, policy, env, guide=None):
        self.policy = policy
        self.env = env
        self.guide = guide

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def begin_episode(self):
        if self.guide is not None:
            self.guide.previous = None

    def predict(self, observation, context, *, action_seed):
        import time
        device = self.policy.system.bc_runtime.device
        if self.guide is not None:
            cloud = torch.as_tensor(observation['point_cloud'][-1], device=device, dtype=torch.float32)
            self.guide.begin(cloud, self.env._drone_pos, float(self.env._drone_yaw))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        bundle = self.policy.predict(observation, context, action_seed=action_seed)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        if self.guide is not None:
            # The memory is the FINAL output, not a denoising intermediate.
            position = torch.as_tensor(self.env._drone_pos, device=bundle.full.device, dtype=torch.float32)
            positions = []
            for delta in bundle.full[0, 2:16, :3]:
                position = position + delta
                positions.append(position)
            self.guide.previous = torch.stack(positions).detach()
        self.last_probe_metrics = dict(inference_seconds=seconds,
            guided=bool(self.guide and self.guide.triggered_steps),
            triggered_ddim_indices=[] if self.guide is None else self.guide.triggered_steps,
            gradient_updates=0 if self.guide is None else self.guide.update_count)
        return bundle
