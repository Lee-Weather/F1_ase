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
                                   standing_joint_groups, standing_pose_mse, velocity_task_reward)


class VelocityTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "data/cfg/humanoid_f1_hlc_velocity.yaml"
        self.cfg = yaml.safe_load(path.read_text(encoding="utf-8"))["env"]["hlc"]
        self.cfg.update(standingSpeedSigma=.05, standFootSpeedScale=2., standLegVelocityScale=.05)

    def test_heading_projection_uses_horizontal_axes(self):
        actual = heading_velocity(np.float32(0), np.float32(.8), -2., 3., .4)
        np.testing.assert_allclose(actual, [3., 2., .4], atol=1e-6)

    def test_forward_yaw_curriculum_has_no_lateral_or_backward_commands(self):
        path = Path(__file__).resolve().parents[1] / "data/cfg/humanoid_f1_hlc_forward_yaw.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["env"]["hlc"]
        torch.manual_seed(20261124)
        for stage in config["commandCurriculum"]:
            command, groups = sample_velocity_commands(stage, 4000, "cpu")
            self.assertTrue((command[:, 0] >= 0.).all())
            self.assertTrue((command[:, 0] <= .5).all())
            self.assertTrue((command[:, 1] == 0.).all())
            self.assertTrue((command[:, 2].abs() <= .2).all())
            self.assertTrue((command[:, 2] == -.2).any() and (command[:, 2] == .2).any())
            self.assertTrue(torch.equal((command == 0).all(-1), groups == 0))
            mixed = command[groups == 7]
            self.assertTrue((mixed[:, 0] > 0).all())
            self.assertTrue((mixed[:, 2] != 0).all())
        self.assertTrue(((command[:, 0] > 0) & (command[:, 0] < .05)).any())

    def test_forward_lateral_yaw_reaches_signed_limits_without_backward(self):
        path = Path(__file__).resolve().parents[1] / "data/cfg/humanoid_f1_hlc_forward_lateral_yaw.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))["env"]["hlc"]
        torch.manual_seed(20261125)
        for epoch, stage_index in ((1, 1), (200, 1), (201, 2), (600, 2), (601, 3), (1800, 3)):
            with self.subTest(epoch=epoch):
                index, stage = command_stage(config, epoch)
                self.assertEqual(index, stage_index)
                command, groups = sample_velocity_commands(stage, 8192, "cpu")
                vx, vy, wz = command.unbind(-1)
                self.assertTrue(((vx >= 0) & (vx <= .50) & (vy.abs() <= .40) & (wz.abs() <= .40)).all())
                self.assertFalse((groups == 2).any())
                self.assertTrue(torch.equal((command == 0).all(-1), groups == 0))
                self.assertTrue((vx == .50).any())
                for axis in (vy, wz):
                    self.assertTrue((axis == -.4).any() and (axis == .4).any())
                mixed = command[groups == 7]
                expected = stage.get("mixedAxes", [])
                self.assertEqual({tuple(row) for row in (mixed != 0).long().tolist()}, {tuple(row) for row in expected})
                self.assertGreaterEqual(stage["maxPlanarSpeed"], .64032)

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

    def test_standing_pose_rewards_exact_stop_and_respects_braking(self):
        config = dict(self.cfg, standingPoseWeight=.4, standingPoseSigmaRad=[.12,.18,.25],
                      standingPoseGroupWeights=[.30,.45,.25])
        command = np.array([[0,0,0], [0,0,.2], [.35,0,0], [0,0,1e-7]], dtype=np.float32)
        actual = command.copy()
        stationary = np.zeros(len(command), dtype=np.float32)
        groups = standing_joint_groups(["lumbar_pitch_joint", "left_knee_pitch_joint", "left_elbow_pitch_joint"])
        reference = np.zeros(3, dtype=np.float32)
        crouched = np.tile(np.array([.2,.4,.6], dtype=np.float32), (len(command), 1))
        errors = standing_pose_mse(crouched, reference, groups)
        legacy = velocity_task_reward(actual, command, stationary, stationary, self.cfg)
        shaped = velocity_task_reward(actual, command, stationary, stationary, config, errors)
        self.assertLess(shaped[0], legacy[0])
        np.testing.assert_array_equal(shaped[1:], legacy[1:])
        good = velocity_task_reward(actual, command, stationary, stationary, config,
                                    standing_pose_mse(np.zeros_like(crouched), reference, groups))
        np.testing.assert_array_equal(good, legacy)
        actual[0, 0] = .1
        moving = velocity_task_reward(actual, command, stationary, stationary, config, errors)
        self.assertLess(moving[0], shaped[0])
        self.assertLess(1 - moving[0] / velocity_task_reward(actual, command, stationary, stationary, self.cfg)[0],
                        1 - shaped[0] / legacy[0])
        torch_reward = velocity_task_reward(torch.from_numpy(actual), torch.from_numpy(command),
                                           torch.from_numpy(stationary), torch.from_numpy(stationary), config,
                                           tuple(torch.from_numpy(value) for value in errors))
        np.testing.assert_allclose(torch_reward.numpy(), moving, atol=1e-6)

    def test_pose_metrics_use_only_valid_quiet_zero_commands(self):
        metrics = VelocityMetrics("cpu", standing_pose=True)
        groups = torch.tensor([0, 0, 0, 0, 5])
        actual = torch.zeros((5, 3))
        command = actual.clone()
        command[-1, 2] = .2
        valid = torch.tensor([True, True, True, False, True])
        quiet = torch.tensor([True, False, True, True, True])
        failed = torch.tensor([False, False, True, False, False])
        pose = torch.tensor([.01, 1., 1., 1., 1.])
        metrics.step(groups, actual, command, valid, quiet, failed, (pose, pose * 4, pose * 9))
        summary = metrics.summarize(metrics.drain())
        self.assertEqual(summary["HLC_Velocity/stop/pose_samples"], 1.)
        self.assertAlmostEqual(summary["HLC_Velocity/stop/pose_legs_rms_deg"], 11.459156, places=5)
