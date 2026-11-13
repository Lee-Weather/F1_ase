"""Frozen F1 ASE inference and the shared HLC control interval."""
import hashlib
import inspect
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.f1_control import checkpoint_control


def unit_latent(value):
    norm = value.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(value)
    fallback[..., 0] = 1
    return torch.where(norm > 1e-8, value / norm.clamp(min=1e-8), fallback)


class FrozenF1LLC(nn.Module):
    """Actor + discriminator only; weights and normalizers are frozen buffers.

    No LLC optimizer, critic, encoder, replay buffer, or environment state is
    restored. HLC optimization cannot discover these weights as parameters.
    """
    def __init__(self, checkpoint, device):
        super().__init__()
        path = Path(checkpoint)
        kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
        saved = torch.load(str(path), map_location=device, **kwargs)
        self._weight_names = {}
        for key, value in saved["model"].items():
            name = key[len("a2c_network."):]
            if name.startswith(("actor_mlp.", "mu.", "_disc_mlp.", "_disc_logits.")):
                buffer = name.replace(".", "__")
                self.register_buffer(buffer, value.detach().float().clone())
                self._weight_names[name] = buffer
        if "actor_mlp._style_dense.weight" not in self._weight_names:
            raise ValueError("F1 HLC requires a pretrained ASE LLC")
        for prefix, key in (("actor", "running_mean_std"), ("amp", "amp_input_mean_std")):
            for stat, value in saved[key].items():
                self.register_buffer(prefix + "_" + stat, value.detach().clone())
        if self.actor_running_mean.numel() != 93 or self.amp_running_mean.numel() != 890:
            raise ValueError("Expected F1 actor93 and AMP890")
        self.control_config = checkpoint_control(saved)
        self.metadata = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             epoch=int(saved["epoch"]), frame=int(saved.get("frame", 0)),
                             actor_dim=93, amp_dim=890, latent_dim=32, action_dim=29)
        self.to(device)
        self.eval()

    def _linear(self, x, name):
        return F.linear(x, getattr(self, self._weight_names[name+".weight"]),
                        getattr(self, self._weight_names[name+".bias"]))

    def _normalize(self, x, prefix):
        mean = getattr(self, prefix+"_running_mean").float()
        var = getattr(self, prefix+"_running_var").float()
        return ((x.float()-mean) / torch.sqrt(var+1e-5)).clamp(-5, 5)

    @torch.no_grad()
    def forward(self, obs, z):
        obs = self._normalize(obs, "actor")
        style = F.relu(self._linear(z, "actor_mlp._style_mlp.0"))
        style = F.relu(self._linear(style, "actor_mlp._style_mlp.2"))
        style = torch.tanh(self._linear(style, "actor_mlp._style_dense"))
        hidden = torch.cat((obs, style), -1)
        for i in range(3):
            hidden = F.relu(self._linear(hidden, "actor_mlp._dense_layers."+str(i)))
        return self._linear(hidden, "mu").clamp(-1, 1)

    @torch.no_grad()
    def discriminator(self, amp):
        hidden = self._normalize(amp, "amp")
        for i in (0, 2, 4):
            hidden = F.relu(self._linear(hidden, "_disc_mlp."+str(i)))
        return self._linear(hidden, "_disc_logits")

    @torch.no_grad()
    def disc_reward(self, amp):
        probability = torch.sigmoid(self.discriminator(amp))
        return -torch.log((1-probability).clamp(min=1e-4)) * 2


@torch.no_grad()
def control_interval(task, llc, z, llc_steps, step=None, on_step=None):
    """One HLC action; latch first terminal states and exclude later substeps.

    Surviving environments finish the interval. A terminated environment is not
    reset inside it, and contributes no reward or telemetry after termination.
    """
    z = unit_latent(z)
    count = task.num_envs
    active = torch.ones(count, dtype=torch.bool, device=task.device)
    done = torch.zeros_like(active)
    terminated = torch.zeros_like(active)
    reward = torch.zeros(count, device=task.device)
    disc = torch.zeros_like(reward)
    samples = torch.zeros_like(reward)
    result_obs, result_states = task.obs_buf.clone(), task.states_buf.clone()
    reasons, metrics = {}, {}
    terminal_steps = torch.zeros(count, dtype=torch.long, device=task.device)
    for index in range(llc_steps):
        action = llc(task.obs_buf[:, :93], z)
        action[~active] = 0
        if step is None:
            task.step(action)
            task.gym.fetch_results(task.sim, True)
        else:
            step(action)
        valid = active.clone()
        reward += task.rew_buf * valid
        disc += llc.disc_reward(task.extras["amp_obs"]).squeeze(-1) * valid
        samples += valid
        result_obs[valid] = task.obs_buf[valid]
        result_states[valid] = task.states_buf[valid]
        terminal_steps[valid] = index+1
        for name, mask in task.extras["termination_reasons"].items():
            reasons[name] = reasons.get(name, torch.zeros_like(active)) | (mask & valid)
        for name, value in task.extras["motion_metrics"].items():
            if name not in metrics:
                metrics[name] = value.clone()
            metrics[name][valid] = value[valid]
        terminated |= task.extras["terminate"].bool() & valid
        done |= task.reset_buf.bool() & valid
        active &= ~done
        if on_step is not None:
            on_step(task, z, action)
        if not active.any():
            break
    info = dict(terminate=terminated.long(), termination_reasons=reasons, motion_metrics=metrics,
                disc_rewards=(disc/samples.clamp(min=1)).unsqueeze(-1), task_rewards=reward/samples.clamp(min=1),
                llc_substeps=terminal_steps)
    return {"obs": result_obs, "states": result_states}, info["task_rewards"], done.long(), info


def load_hlc_actor(checkpoint, llc, device):
    """Deterministic ReLU HRL actor; critic is unnecessary for recorded replay."""
    kwargs = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    saved = torch.load(str(checkpoint), map_location=device, **kwargs)
    if saved.get("llc_dependency") != llc.metadata:
        raise ValueError("HLC checkpoint requires a different LLC")
    weights = {k[len("a2c_network."):]: v.detach() for k, v in saved["model"].items()}
    stats = saved["running_mean_std"]
    mean, var = stats["running_mean"].float(), stats["running_var"].float()
    layers = sorted(int(k.split(".")[1]) for k in weights
                    if k.startswith("actor_mlp.") and k.endswith(".weight"))

    @torch.no_grad()
    def actor(obs):
        hidden = ((obs-mean)/torch.sqrt(var+1e-5)).clamp(-5, 5)
        for i in layers:
            name = "actor_mlp."+str(i)
            hidden = F.relu(F.linear(hidden, weights[name+".weight"], weights[name+".bias"]))
        return torch.tanh(F.linear(hidden, weights["mu.weight"], weights["mu.bias"]))
    return actor, saved
