"""CPU checks for velocity control. No simulator, optimizer step or server required.

Run: python -m unittest discover -s ase/tests -v
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

ASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASE))
from utils.f1_velocity import (encoder_command_mask, encoder_motion_mask, load_env_config, masked_mean,
                               policy_spec, ramp_commands, scripted_speed, task_rewards,
                               velocity_settings)


class VelocityControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.env = load_env_config(ASE / "data/cfg/humanoid_f1_velocity.yaml")
        cls.settings = velocity_settings(cls.env["env"])

    def test_command_ramp_matches_numpy_and_stops(self):
        self.assertEqual(scripted_speed(.49, "walk"), 0)
        self.assertEqual(scripted_speed(14.0, "walk"), .5)
        current = np.zeros((2, 3), dtype=np.float32)
        tensor = torch.from_numpy(current.copy())
        for step in range(2250):
            target = np.array([[scripted_speed(step * .02), 0, 0], [0, 0, 0]], dtype=np.float32)
            next_command = ramp_commands(current, target, .02, self.settings)
            tensor = ramp_commands(tensor, torch.from_numpy(target), .02, self.settings)
            np.testing.assert_allclose(tensor.numpy(), next_command, atol=1e-7)
            self.assertLessEqual(float((next_command - current)[:, 0].max()), .0060001)
            self.assertGreaterEqual(float((next_command - current)[:, 0].min()), -.0100001)
            current = next_command
        np.testing.assert_array_equal(current, 0)

    def test_encoder_gate_has_no_standing_gradient(self):
        commands = torch.tensor([[0., 0, 0], [.1, 0, 0], [.3, 0, 0]])
        mask = encoder_command_mask(commands, self.settings).unsqueeze(-1)
        torch.testing.assert_close(mask, torch.tensor([[0.], [.5], [1.]]))
        values = torch.tensor([[2.], [4.], [6.]], requires_grad=True)
        loss = masked_mean(values, mask)
        loss.backward()
        self.assertEqual(values.grad[0].item(), 0)
        self.assertAlmostEqual(loss.item(), 16 / 3, places=5)
        empty = torch.randn(4, 1, requires_grad=True)
        zero = masked_mean(empty, torch.zeros_like(empty))
        zero.backward()
        self.assertEqual(zero.item(), 0)
        self.assertTrue(torch.equal(empty.grad, torch.zeros_like(empty)))
        # A moving command or a brief/invalid motion history alone cannot open it.
        forward = torch.tensor([0., .15, .15])
        actual = encoder_motion_mask(commands, forward, torch.tensor([True, False, True]), self.settings)
        torch.testing.assert_close(actual, torch.tensor([0., 0., 1.]))
        self.assertEqual(encoder_motion_mask(commands, torch.zeros(3), torch.ones(3), self.settings).sum(), 0)

    def test_tracking_and_stand_rewards_have_correct_targets(self):
        velocity = torch.tensor([[0., 0, 0], [.6, 0, 0], [0., 0, 0], [.6, 0, 0]])
        commands = torch.tensor([[0., 0, 0], [.6, 0, 0], [.6, 0, 0], [0., 0, 0]])
        zeros = torch.zeros(4)
        joints = torch.zeros(4, 29)
        posture = dict(foot_yaw=torch.zeros(4, 2), hip_yaw=torch.zeros(4, 2), foot_tilt=torch.zeros(4, 2),
                       stance=torch.ones(4, 2), stand_pose_error=zeros, pd_targets=joints)
        reward, terms = task_rewards(velocity, zeros, zeros, torch.zeros(4, 12), torch.zeros(4, 2),
            joints, joints, joints, torch.ones(29), joints, -torch.ones(29), torch.ones(29),
            commands, self.settings, posture)
        self.assertGreater(reward[1].item(), reward[2].item())
        self.assertGreater(reward[0].item(), reward[3].item())
        self.assertGreater(terms["stand_quiet"][0].item(), 0)
        self.assertEqual(terms["stand_quiet"][3].item(), 0)
        torch.testing.assert_close(reward, sum(terms.values()))
        self.assertTrue(torch.isfinite(reward).all())
        # 90-degree feet are worse at identical speed and commands; a foot in
        # flight is not punished for sole tilt. Limit cost also sees PD targets.
        posture["foot_yaw"][:] = np.pi / 2
        posture["foot_tilt"][:] = np.pi / 2
        posture["stance"][:] = 0
        posture["pd_targets"] = torch.full_like(joints, 2)
        sideways, costs = task_rewards(velocity, zeros, zeros, torch.zeros(4, 12), torch.zeros(4, 2),
            joints, joints, joints, torch.ones(29), joints, -torch.ones(29), torch.ones(29),
            commands, self.settings, posture)
        self.assertTrue((sideways < reward).all())
        self.assertTrue((costs["foot_heading"] < 0).all())
        self.assertTrue((costs["joint_limits"] < 0).all())
        self.assertTrue(costs["stance_foot_tilt"].eq(0).all())
        self.assertAlmostEqual(float(terms["lin_vel_tracking"][2]), .4, places=5)

    def test_target_windows_cover_references_and_legacy_mapping(self):
        import xml.etree.ElementTree as ET
        from replay_f1 import joint_names
        from utils.f1_control import pd_action_parameters, action_to_targets, checkpoint_control
        names = joint_names()
        path = ASE / "data/assets/f1/urdf/f1_29dof.urdf"
        joints = {joint.get("name"): joint.find("limit") for joint in ET.parse(path).getroot().findall("joint")}
        lower = np.array([float(joints[name].get("lower")) for name in names], dtype=np.float32)
        upper = np.array([float(joints[name].get("upper")) for name in names], dtype=np.float32)
        original = lower.copy(), upper.copy()
        neutral, negative, positive = pd_action_parameters(self.env["env"], names, lower, upper)
        folder = ASE / "data/motions/f1"
        entries = yaml.safe_load((folder / "dataset_f1_walk_stop.yaml").read_text())["motions"]
        for name, limits in self.env["env"]["pdTargetLimits"].items():
            index = names.index(name)
            self.assertAlmostEqual(float(action_to_targets(-np.ones(29), neutral, negative, positive)[index]), limits[0], places=6)
            self.assertAlmostEqual(float(action_to_targets(np.ones(29), neutral, negative, positive)[index]), limits[1], places=6)
            for entry in entries:
                with np.load(folder / entry["file"]) as data:
                    q = data["joint_pos"][:, data["joint_names"].tolist().index(name)]
                    self.assertGreaterEqual(float(q.min()), limits[0])
                    self.assertLessEqual(float(q.max()), limits[1])
        np.testing.assert_array_equal(lower, original[0])
        np.testing.assert_array_equal(upper, original[1])
        old_env = dict(self.env["env"], **checkpoint_control({"control_config": {"actionMapping": "joint_limits"}}))
        old = pd_action_parameters(old_env, names, lower, upper)
        hip = names.index("left_hip_yaw_joint")
        self.assertGreater(float(action_to_targets(np.ones(29), *old)[hip]), 1.57)
        self.assertEqual(checkpoint_control({})["pdTargetLimits"], {})

    def test_configuration_inherits_control_and_reuses_motions(self):
        baseline = load_env_config(ASE / "data/cfg/humanoid_f1.yaml")
        for name in ("asset", "pdGains", "defaultJointPos", "armature", "actionMapping", "actionLimitMargin"):
            self.assertEqual(self.env["env"][name], baseline["env"][name])
        self.assertEqual(self.env["env"]["episodeLength"], 1500)
        self.assertEqual(baseline["env"]["episodeLength"], 500)
        folder = ASE / "data/motions/f1"
        motions = yaml.safe_load((folder / "dataset_f1_walk_stop.yaml").read_text())["motions"]
        self.assertAlmostEqual(sum(item["weight"] for item in motions), 1)
        self.assertEqual(len(motions), 5)
        self.assertTrue(all((folder / item["file"]).is_file() for item in motions))

    def test_real_ase_network_and_portable_checkpoint(self):
        from learning.ase_network_builder import ASEBuilder
        from learning.amp_datasets import AMPDataset
        from replay_f1 import export_actor_npz, load_actor, load_actor_numpy
        cfg = yaml.safe_load((ASE / "data/cfg/train/rlg/ase_f1_velocity.yaml").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            net = ASEBuilder.Network(cfg["params"]["network"], actions_num=29, input_shape=(96,),
                critic_input_shape=(451,), amp_input_shape=(890,), ase_latent_shape=(32,),
                normalize_critic_input=True)
        obs, states, amp = torch.randn(8, 96), torch.randn(8, 451), torch.randn(8, 890)
        z = net.sample_latents(8)
        action, _ = net.eval_actor(obs, z)
        value = net.eval_critic(states, z)
        encoded = net.eval_enc(amp)
        self.assertEqual(action.shape, (8, 29))
        self.assertEqual(value.shape, (8, 1))
        self.assertEqual(encoded.shape, (8, 32))
        loss = masked_mean(-(encoded * z).sum(-1, keepdim=True), torch.ones(8, 1)) + action.square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
        # Identical flatten/shuffle on observations and their per-transition masks.
        raw = torch.arange(8, dtype=torch.float32).reshape(2, 4, 1)
        data = AMPDataset(8, 4, False, False, "cpu", 1)
        flat = raw.transpose(0, 1).reshape(-1, 1)
        data.values_dict = {"obs": flat.repeat(1, 96), "enc_reward_mask": flat}
        for i in range(2):
            batch = data[i]
            torch.testing.assert_close(batch["obs"][:, :1], batch["enc_reward_mask"])
        spec = policy_spec(self.settings, .02, "ase/data/motions/f1/dataset_f1_walk_stop.yaml")
        self.assertEqual(spec["actor_obs"], 96)
        self.assertEqual(sum(size for _, size in spec["actor_fields"]), 96)
        control = {name: self.env["env"][name] for name in ("actionMapping", "pdGains", "defaultJointPos", "pdTargetLimits")}
        checkpoint = dict(epoch=0, frame=0, control_config=control, policy_config=spec,
            running_mean_std={"running_mean": torch.zeros(96), "running_var": torch.ones(96)},
            model={"a2c_network."+name: tensor.detach() for name, tensor in net.state_dict().items()})
        with tempfile.TemporaryDirectory(prefix="f1-velocity-policy-") as temporary:
            model, portable = Path(temporary)/"model.pt", Path(temporary)/"actor.npz"
            torch.save(checkpoint, model)
            export_actor_npz(model, portable)
            with contextlib.redirect_stdout(io.StringIO()):
                actor = load_actor(model)
            numpy_actor = load_actor_numpy(portable)
            actual = actor(obs, z).numpy()
            expected = net.eval_actor(obs / np.sqrt(1 + 1e-5), z)[0].clamp(-1, 1).detach().numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-6)
            np.testing.assert_allclose(numpy_actor(obs.numpy(), z.numpy()), actual, atol=3e-6)
            self.assertEqual(actor.policy_config, spec)
            self.assertEqual(numpy_actor.policy_config, spec)

    def test_replay_summary_measures_a_continuous_stop(self):
        from replay_f1 import velocity_summary
        command = np.zeros((1, 3), dtype=np.float32)
        rows, x = [], 0.0
        for step in range(2250):
            target = scripted_speed(step * .02)
            command = ramp_commands(command, np.array([[target, 0, 0]]), .02, self.settings)
            c = float(command[0, 0])
            x += c * .02
            rows.append([(step+1)*.02, target, c, c, 0, 0, 0, 1, 0, 0, 0, x, 0])
        result = velocity_summary(rows)
        self.assertTrue(result["numeric_targets_met"], result)
        # A quiet mean foot speed must not hide one foot still moving.
        feet_max = np.zeros(len(rows))
        feet_max[-250:] = .11
        self.assertFalse(velocity_summary(rows, feet_max)["numeric_targets_met"])
        # A robot that keeps walking after the zero target must fail.
        for row in rows[-250:]:
            row[3] = .3
        self.assertFalse(velocity_summary(rows)["numeric_targets_met"])

    def test_amp_schema_and_raw_knee_penalty(self):
        from utils.f1_control import checkpoint_control
        env = load_env_config(ASE / "data/cfg/humanoid_f1_velocity_amp.yaml")["env"]
        settings = velocity_settings(env)
        spec = policy_spec(settings, .02, "motions.yaml", "AMP")
        self.assertEqual((spec["version"], spec["task_reward_version"], spec["latent_dim"]), (3, 3, 0))
        self.assertNotIn("latent_hold_episode", spec)
        self.assertFalse(any(key.startswith("encoder") for key in settings))
        self.assertEqual(env["velocityDefaultStage"], "B")
        self.assertEqual(checkpoint_control({})["pdTargetClampJoints"], [])
        raw = torch.zeros(1, 29, requires_grad=True)
        with torch.no_grad():
            raw[0, :2] = -.2
        q = torch.zeros(1, 29)
        posture = dict(foot_yaw=torch.zeros(1, 2), hip_yaw=torch.zeros(1, 2),
                       foot_tilt=torch.zeros(1, 2), stance=torch.ones(1, 2),
                       stand_pose_error=torch.zeros(1), pd_targets=raw, knee_ids=[0, 1])
        _, terms = task_rewards(torch.zeros(1, 3), torch.zeros(1), torch.zeros(1),
            torch.zeros(1, 12), torch.zeros(1, 2), q, q, q, torch.ones(29), q,
            torch.zeros(29), torch.ones(29), torch.zeros(1, 3), settings, posture)
        self.assertAlmostEqual(float(terms["joint_limits"]), -.2 * .2**2, places=6)
        (-terms["joint_limits"].sum()).backward()
        self.assertTrue((raw.grad[0, :2] < 0).all())

    def test_task_command_and_reward_timing_without_physics(self):
        """Exercise the real task lifecycle with only Gym's I/O boundary replaced."""
        import importlib
        from copy import deepcopy
        from replay_f1 import joint_names

        class BufferTask:
            def __init__(self, cfg, *args):
                self.cfg, self.device, self.num_envs, self.dt = cfg, "cpu", 2, .02
                self._setup_character_props(cfg["env"]["keyBodies"])
                self._dof_names = joint_names()
                self._num_amp_obs_steps = 10
                self.obs_buf = torch.zeros(2, self._num_obs)
                self.states_buf = torch.zeros(2, cfg["env"]["numStates"])
                for name in ("actions", "_dof_vel", "_dof_pos", "dof_force_tensor"):
                    setattr(self, name, torch.zeros(2, 29))
                self.motor_efforts = self.dof_limits_upper = torch.ones(29)
                self.dof_limits_lower = -torch.ones(29)
                self.rew_buf = torch.zeros(2)
                self.progress_buf = torch.zeros(2, dtype=torch.long)
                self._terminate_buf = torch.zeros(2, dtype=torch.long)
                self.extras = {}

            def _setup_character_props(self, bodies):
                self._num_obs = 93
                self.cfg["env"]["numStates"] = 448

            def _build_key_body_ids_tensor(self, names):
                return torch.arange(len(names))

            def _compute_observations(self, env_ids=None):
                pass

            def _compute_reset(self):
                self.extras["motion_metrics"] = {}

            def _reset_envs(self, ids):
                self.progress_buf[ids] = 0
                self.actions[ids] = 0
                self._compute_observations(ids)

            def pre_physics_step(self, actions):
                self.actions = actions.clone()

            def post_physics_step(self):
                self.progress_buf += 1
                self._compute_observations()
                self._compute_reward(self.actions)
                self._compute_reset()

        base = types.ModuleType("env.tasks.humanoid_f1")
        base.HumanoidF1 = BufferTask
        rotations = types.ModuleType("isaacgym.torch_utils")
        rotations.quat_rotate = rotations.quat_rotate_inverse = lambda q, v: v
        modules = {"env.tasks.humanoid_f1": base, "isaacgym": types.ModuleType("isaacgym"),
                   "isaacgym.torch_utils": rotations, "utils.torch_utils": types.ModuleType("utils.torch_utils")}
        module_name = "env.tasks.humanoid_f1_velocity"
        # Preserve the v1 timing contract too. v2 physical displacement and
        # curriculum timing are covered by the small real-Gym smoke check.
        legacy = deepcopy(self.env)
        legacy["env"]["velocityTask"].pop("version")
        with mock.patch.dict(sys.modules, modules):
            module = importlib.import_module(module_name)
            task = module.HumanoidF1Velocity(legacy, None, None, "cpu", 0, True)
        sys.modules.pop(module_name, None)
        task._motion_state = lambda: (torch.zeros(2, 3), torch.zeros(2), torch.zeros(2), torch.zeros(2, 2))
        task._reset_envs(torch.arange(2))
        task._command_steps.fill_(1)
        for _ in range(40):
            seen = task.obs_buf[:, 93:].clone()
            progress = task.progress_buf.clone()
            task.pre_physics_step(torch.zeros(2, 29))
            task.post_physics_step()
            torch.testing.assert_close(task._reward_commands, seen)
            expected = encoder_command_mask(seen, self.settings) * (progress >= 9)
            torch.testing.assert_close(task.extras["enc_reward_mask"][:, 0], expected)
            torch.testing.assert_close(task.obs_buf[:, 93:], task.commands)
            torch.testing.assert_close(task.states_buf[:, 448:], task.commands)
        task.set_velocity_target(.8)
        obs = task.prepare_velocity_command().clone()
        applied = obs[:, 93:].clone()
        task.pre_physics_step(torch.zeros(2, 29))
        task.post_physics_step()
        torch.testing.assert_close(task.commands, applied)  # No second ramp in external mode.
        self.assertEqual(task.obs_buf.shape, (2, 96))
        self.assertEqual(task.states_buf.shape, (2, 451))


if __name__ == "__main__":
    unittest.main()
