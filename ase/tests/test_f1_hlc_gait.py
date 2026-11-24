"""Regression tests for complete alternating footfalls and stance slip."""

from pathlib import Path
import sys
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.f1_hlc_gait import AlternatingGait
from utils.f1_hlc_metrics import HLCMetrics, summarize_hlc_metrics


class GaitTests(unittest.TestCase):
    def setUp(self):
        cfg_path = Path(__file__).resolve().parents[1] / "data/cfg/humanoid_f1_hlc_alternating_gait.yaml"
        self.config = yaml.safe_load(cfg_path.read_text())["env"]["hlc"]
        self.dt = 0.01999999955
        self.gait = AlternatingGait(1, "cpu", self.dt, self.config["gaitReward"], torch.zeros(2, 1, 3))
        self.position = torch.zeros(1, 2, 3)
        self.rotation = torch.tensor([[[0., 0., 0., 1.], [0., 0., 0., 1.]]])
        self.load = torch.full((1, 2), 70.)
        self.observe()

    def observe(self, moving=True):
        return self.gait.step(self.position, self.rotation, self.load, torch.tensor([moving]))

    def swing(self, side, air_steps=4, clearance=.02, advance=.04, opposite_load=70., stop=False):
        self.load[0, side] = 0
        self.load[0, 1 - side] = opposite_load
        self.position[0, side, 0] += advance
        self.position[0, side, 2] = clearance
        for _ in range(air_steps):
            self.observe()
        if stop:
            self.observe(moving=False)
        self.load.fill_(70.)
        self.position[0, side, 2] = 0
        return self.observe()

    def test_repeated_one_side_never_enables_alternation(self):
        for _ in range(4):
            recent, event, complete, *_ = self.swing(0)
            self.assertEqual(complete.tolist(), [[True, False]])
            self.assertFalse(event.item())
            self.assertEqual(recent.item(), 0)
        recent, event, complete, *_ = self.swing(1)
        self.assertEqual(complete.tolist(), [[False, True]])
        self.assertTrue(event.item())
        self.assertEqual(recent.item(), 1)
        self.assertTrue(self.swing(0)[1].item())
        self.assertFalse(self.swing(0)[1].item())
        self.assertEqual(self.observe()[0].item(), 0)

    def test_incomplete_swings_do_not_count(self):
        for params in ({"air_steps": 3}, {"clearance": .005}, {"advance": .01}, {"opposite_load": 5.}):
            with self.subTest(params=params):
                self.assertFalse(self.swing(0, **params)[2].any().item())

    def test_zero_and_reset_clear_history(self):
        self.swing(0)
        self.assertFalse(self.swing(1, stop=True)[2].any().item())
        self.assertFalse(self.swing(1)[1].item())
        self.assertTrue(self.swing(0)[1].item())
        self.gait.reset(torch.tensor([0]))
        self.assertEqual(self.observe()[0].item(), 0)
        self.assertFalse(self.swing(1)[1].item())

    def test_stance_slip_excludes_swing_and_landing(self):
        self.position[0, 0, 0] += .002
        _, _, _, slip, stance = self.observe()
        self.assertEqual(stance.tolist(), [[True, True]])
        torch.testing.assert_close(slip, torch.tensor([[.1, 0.]]))
        _, _, _, slip, stance = self.swing(0)
        self.assertEqual(stance.tolist(), [[False, True]])
        self.assertEqual(slip.tolist(), [[0., 0.]])

    def test_sliding_reduces_reward_and_telemetry_uses_stance_samples(self):
        self.swing(0)
        recent, event, complete, slip, stance = self.swing(1)
        reward = torch.tensor([.8])
        alternating = self.gait.moving_reward(reward, recent, event, slip, stance)
        no_gait = self.gait.moving_reward(reward, torch.zeros_like(recent), ~event, slip, stance)
        self.assertGreater(alternating.item(), no_gait.item())
        self.position[0, 1, 0] += .004
        recent, event, complete, slip, stance = self.observe()
        sliding = self.gait.moving_reward(reward, recent, event, slip, stance)
        no_slip = self.gait.moving_reward(reward, recent, event, torch.zeros_like(slip), stance)
        self.assertLess(sliding.item(), no_slip.item())
        metrics = HLCMetrics(1, "cpu", self.dt, self.config)
        metrics.record_gait(torch.tensor([True]), recent, event, complete, slip, stance)
        summary = summarize_hlc_metrics({key: value.item() for key, value in metrics.drain().items()})
        self.assertEqual(summary["HLC_Gait/stance_samples"], 2)
        self.assertAlmostEqual(summary["HLC_Gait/stance_slip_mean_m_s"], .1, places=5)


if __name__ == "__main__":
    unittest.main()
