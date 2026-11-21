"""Three-axis frame, sampling, reward, and per-command telemetry regressions."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.f1_hlc_curriculum import command_stage
from utils.f1_hlc_velocity import (VelocityMetrics, heading_velocity, sample_velocity_commands,
                                   velocity_task_reward)


class VelocityTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "data/cfg/humanoid_f1_hlc_velocity.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["env"]["hlc"]
        self.cfg.update(standingSpeedSigma=.05, standFootSpeedScale=2., standLegVelocityScale=.05)

    def test_heading_projection_uses_horizontal_axes(self):
        actual = heading_velocity(np.float32(0), np.float32(.8), -2., 3., .4)
        np.testing.assert_allclose(actual, [3., 2., .4], atol=1e-6)

    def test_curriculum_bounds_and_command_groups(self):
        torch.manual_seed(20261122)
        for epoch, expected in ((1, 1), (200, 1), (201, 2), (600, 2), (601, 3), (1800, 3)):
            index, stage = command_stage(self.cfg, epoch)
            self.assertEqual(index, expected)
            target, groups = sample_velocity_commands(stage, 4000, "cpu")
            self.assertTrue(torch.all(target[:, :2].norm(dim=-1) <= .600001))
            self.assertTrue(torch.all(target >= torch.tensor([-.3, -.25, -.5])))
            self.assertTrue(torch.all(target <= torch.tensor([.6, .25, .5])))
            self.assertTrue(torch.equal((target == 0).all(-1), groups == 0))
            weights = torch.tensor(stage["commandWeights"])
            counts = torch.bincount(groups, minlength=len(weights))
            self.assertTrue(torch.equal(counts > 0, weights > 0))
            torch.testing.assert_close(counts / len(groups), weights, atol=.03, rtol=0.)
            if index == 3:
                self.assertTrue(torch.all((target[groups == 7] != 0).sum(-1) >= 2))

    def test_reward_supports_signed_axes_and_torch_numpy_parity(self):
        commands = np.array([[.4,0,0], [-.25,0,0], [0,.2,0], [0,-.2,0],
                             [0,0,.3], [0,0,-.3], [.25,-.1,.2], [0,0,0]], dtype=np.float32)
        quiet = np.zeros(len(commands), dtype=np.float32)
        best = velocity_task_reward(commands, commands, quiet, quiet, self.cfg)
        stationary = velocity_task_reward(np.zeros_like(commands), commands, quiet, quiet, self.cfg)
        backwards = velocity_task_reward(-commands, commands, quiet, quiet, self.cfg)
        self.assertTrue((best == 1).all())
        self.assertTrue((best[:-1] > stationary[:-1]).all())
        self.assertTrue((stationary[:-1] > backwards[:-1]).all())
        self.assertLess(velocity_task_reward(commands[4], commands[-1], 0., 0., self.cfg), .2)
        noisy = commands + np.float32(.04)
        result = velocity_task_reward(torch.from_numpy(noisy), torch.from_numpy(commands),
                                      torch.from_numpy(quiet), torch.from_numpy(quiet), self.cfg)
        np.testing.assert_allclose(result.numpy(), velocity_task_reward(noisy, commands, quiet, quiet, self.cfg),
                                   atol=1e-6)

    def test_metrics_drop_inactive_samples(self):
        metrics = VelocityMetrics("cpu")
        target = torch.tensor([[0.,0.,0.], [-.2,0.,0.], [-.3,0.,0.]])
        actual = torch.tensor([[0.,0.,0.], [-.1,0.,0.], [1.,0.,0.]])
        metrics.step(torch.tensor([0, 2, 2]), actual, target, torch.tensor([True, True, False]),
                     torch.tensor([True, False, False]), torch.tensor([False, True, True]))
        result = metrics.summarize(metrics.drain())
        self.assertAlmostEqual(result["HLC_Velocity/backward/vx_mae_m_s"], .1)
        self.assertEqual(result["HLC_Velocity/backward/falls"], 1)
        self.assertEqual(result["HLC_Velocity/stop/quiet_fraction"], 1.)
        self.assertEqual(metrics.summarize(metrics.drain()), {})
