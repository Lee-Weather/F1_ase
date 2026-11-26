"""Alternating HLC and output-layer LLC PPO using fresh task rollouts."""

import math
import time

from rl_games.common.a2c_common import swap_and_flatten01
import torch
from torch import nn
from torch.nn import functional as F

from learning.hrl_agent import HRLAgent
from utils.f1_hlc import control_interval


class MixedHRLAgent(HRLAgent):
    def __init__(self, base_name, config):
        super().__init__(base_name, config)
        self._mix = self.vec_env.env.task.hlc["llcFineTune"]
        if (not config.get("frozen_f1_llc") or self.is_rnn or self.multi_gpu
                or self.num_agents != 1 or self.has_central_value):
            raise ValueError("F1 alternating training requires the F1 LLC path and single-GPU feedforward PPO")
        if (self._mix["warmupEpochs"] < 0 or self._mix["updateEvery"] < 2 or self._mix["minibatchSize"] < 1
                or not all(math.isfinite(self._mix[k]) and self._mix[k] > 0
                           for k in ("learningRate", "actionStd", "anchorWeight"))):
            raise ValueError("Invalid LLC fine-tuning schedule or optimizer settings")
        weight, bias = self._llc_agent.mu__weight, self._llc_agent.mu__bias
        self._original_head = {"weight": weight.detach().clone(), "bias": bias.detach().clone()}
        self._llc_head = nn.Linear(weight.shape[1], weight.shape[0], device=weight.device)
        self._llc_head.load_state_dict(self._original_head)
        self._llc_optimizer = torch.optim.Adam(self._llc_head.parameters(), lr=self._mix["learningRate"])
        self._llc_updates = 0
        self._llc_rollout = None

    def _train_llc_this_epoch(self):
        relative = self.epoch_num - self._mix["warmupEpochs"] - 1
        return relative >= 0 and relative % self._mix["updateEvery"] == 0

    def env_step(self, actions):
        if self._llc_rollout is None:
            return super().env_step(actions)
        task = self.vec_env.env.task
        buffers = self._llc_rollout
        interval = self._rollout_interval
        std = self._mix["actionStd"]

        def sample(obs, z, active, substep):
            mean = self._llc_head(self._llc_agent.actor_features(obs, z)).clamp(-1, 1)
            noise = torch.randn_like(mean)
            raw_action = mean + std * noise
            buffers["obs"][interval, :, substep].copy_(obs)
            buffers["action"][interval, :, substep].copy_(raw_action)
            buffers["valid"][interval, :, substep].copy_(active)
            buffers["logp"][interval].add_(-0.5 * noise.square().sum(dim=-1) * active)
            if substep == 0:
                buffers["z"][interval].copy_(z)
            return raw_action.clamp(-1, 1)

        obs, rewards, dones, info = control_interval(
            task, self._llc_agent, self.preprocess_actions(actions), self._llc_steps,
            step=self.vec_env.step, sample_action=sample)
        self._rollout_interval += 1
        return self.obs_to_tensors(obs), rewards.unsqueeze(-1), dones, info

    def train_epoch(self):
        if not self._train_llc_this_epoch():
            info = super().train_epoch()
            info["mixed_metrics"] = {"Mix/update_llc": 0, "Mix/llc_updates": self._llc_updates}
            return info

        start = time.time()
        shape = (self.horizon_length, self.num_actors)
        dtype = dict(device=self.ppo_device)
        rollout = {
            "obs": torch.zeros((*shape, self._llc_steps, 93), **dtype),
            "action": torch.zeros((*shape, self._llc_steps, 29), **dtype),
            "valid": torch.zeros((*shape, self._llc_steps), dtype=torch.bool, **dtype),
            "z": torch.zeros((*shape, self._latent_dim), **dtype),
            "logp": torch.zeros(shape, **dtype),
        }
        self._llc_rollout, self._rollout_interval = rollout, 0
        try:
            with torch.no_grad():
                batch = self.play_steps()
        finally:
            self._llc_rollout = None
        sampled = time.time()
        self.curr_frames = batch.pop("played_frames")
        self.algo_observer.after_steps()
        stats = self._update_llc(rollout, self._calc_advs(batch))
        self._llc_agent.set_output_head(self._llc_head.state_dict())
        self._llc_updates += 1
        ended = time.time()
        info = dict(play_time=sampled - start, update_time=ended - sampled, total_time=ended - start,
                    actor_loss=stats["actor_loss"], actor_clip_frac=stats["clip_fraction"], kl=stats["kl"],
                    critic_loss=[torch.zeros((), **dtype)], b_loss=[torch.zeros((), **dtype)],
                    entropy=[torch.tensor(29 * (math.log(self._mix["actionStd"]) + 0.5 * math.log(2 * math.pi * math.e)), **dtype)],
                    last_lr=[self._mix["learningRate"]], lr_mul=[1.0])
        info["mixed_metrics"] = {
            "Mix/update_llc": 1, "Mix/llc_updates": self._llc_updates,
            "Mix/llc_action_rmse": torch.stack(stats["anchor_mse"]).mean().sqrt().item(),
            "Mix/llc_learning_rate": self._mix["learningRate"],
            "Mix/llc_action_std": self._mix["actionStd"],
        }
        self._record_train_batch_info(batch, info)
        return info

    def _update_llc(self, rollout, advantages):
        # The held z and the sequence of executed LLC actions form one 10 Hz
        # transition. Its action ratio uses the product of valid 50 Hz densities.
        data = {key: swap_and_flatten01(value) for key, value in rollout.items()}
        std = self._mix["actionStd"]
        stats = {key: [] for key in ("actor_loss", "clip_fraction", "kl", "anchor_mse")}
        order = torch.randperm(len(advantages), device=self.ppo_device)
        for idx in order.split(self._mix["minibatchSize"]):
            valid = data["valid"][idx]
            with torch.no_grad():
                features = self._llc_agent.actor_features(data["obs"][idx], data["z"][idx, None, :].expand(-1, self._llc_steps, -1))
                original = F.linear(features, **self._original_head).clamp(-1, 1)
            mean = self._llc_head(features).clamp(-1, 1)
            logp = -0.5 * ((data["action"][idx] - mean) / std).square().sum(dim=-1)
            log_ratio = (logp * valid).sum(dim=-1) - data["logp"][idx]
            ratio = log_ratio.clamp(-20, 20).exp()
            adv = advantages[idx]
            actor_loss = -torch.minimum(adv * ratio, adv * ratio.clamp(1 - self.e_clip, 1 + self.e_clip)).mean()
            anchor = (mean - original).square().mean(dim=-1)
            anchor_mse = (anchor * valid).sum() / valid.sum().clamp_min(1)
            self._llc_optimizer.zero_grad(set_to_none=True)
            (actor_loss + self._mix["anchorWeight"] * anchor_mse).backward()
            nn.utils.clip_grad_norm_(self._llc_head.parameters(), self.grad_norm)
            self._llc_optimizer.step()
            stats["actor_loss"].append(actor_loss.detach())
            stats["clip_fraction"].append(((ratio.detach() - 1).abs() > self.e_clip).float().mean())
            stats["kl"].append((ratio.detach() - 1 - log_ratio.detach()).mean())
            stats["anchor_mse"].append(anchor_mse.detach())
        return stats

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        for key, value in train_info["mixed_metrics"].items():
            self.writer.add_scalar(key, value, frame)

    def get_full_state_weights(self):
        weights = super().get_full_state_weights()
        weights["llc_finetune"] = dict(head=self._llc_head.state_dict(), optimizer=self._llc_optimizer.state_dict(),
                                      updates=self._llc_updates, config=self._mix.copy())
        return weights

    def set_full_state_weights(self, weights):
        fine_tune = weights.get("llc_finetune")
        if fine_tune is None or fine_tune["config"] != self._mix:
            raise ValueError("Resume requires a matching F1 alternating-training checkpoint")
        super().set_full_state_weights(weights)
        self._llc_head.load_state_dict(fine_tune["head"])
        self._llc_optimizer.load_state_dict(fine_tune["optimizer"])
        self._llc_updates = fine_tune["updates"]
        self._llc_agent.set_output_head(fine_tune["head"])
