"""CPU-only regression checks for event denominators and physical sample counts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from utils.f1_hlc_metrics import HLCMetrics, summarize_hlc_metrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = HLCMetrics(1, "cpu", .1, {})

    def step(self, target, vx, steps=1, foot=.01, failed=False, done=False):
        for _ in range(steps):
            self.metrics.step(torch.tensor([float(target)]), torch.tensor([float(vx)]),
                              torch.zeros(1), torch.tensor([float(foot)]), torch.tensor([.1]),
                              torch.tensor([True]), torch.tensor([failed]), torch.tensor([done]))

    def totals(self):
        return {key: float(value) for key, value in self.metrics.drain().items()}

    def test_float32_control_dt_keeps_exact_windows(self):
        metrics = HLCMetrics(1, "cpu", .019999999552965164, {})
        self.assertEqual((metrics.quiet_steps, metrics.tail_steps,
                          metrics.restart_steps, metrics.deadline_steps), (25, 250, 100, 150))

    def test_stop_restart_and_rollout_boundary(self):
        rng = torch.get_rng_state().clone()
        self.step(.5, .5)
        self.step(0, 0, 20)
        first = self.totals()
        self.assertEqual(first["stop/requested"], 1)
        self.assertEqual(first["stop/completed"], 0)
        self.assertEqual(self.metrics.pending()["stop"], 1)
        self.step(0, 0, 30)
        self.step(.2, .2, 50)
        self.step(.5, .5)
        totals = self.totals()
        self.assertEqual(totals["stop/successes"], 1)
        self.assertEqual(totals["restart/successes"], 1)
        self.assertAlmostEqual(totals["stop/success_delay_sum_s"], .5)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_quiet_snapshot_does_not_override_drift(self):
        self.step(.5, .5)
        self.step(0, 0, 6)
        self.step(0, .2, 44, foot=1.)
        self.step(.3, .3)
        totals = self.totals()
        self.assertEqual(totals["stop/completed"], 1)
        self.assertEqual(totals["stop/successes"], 0)
        self.assertEqual(totals["restart/unqualified"], 1)

    def test_late_quiet_fails_even_when_final_path_is_small(self):
        self.step(.5, .5)
        self.step(0, .2, 35, foot=1.)
        self.step(0, 0, 60)
        self.step(.3, .3)
        totals = self.totals()
        self.assertEqual(totals["stop/completed"], 1)
        self.assertEqual(totals["stop/successes"], 0)

    def test_timeout_and_reset_are_censored_but_fall_fails(self):
        self.step(.5, .5)
        self.step(0, 0, 3)
        self.step(0, 0, done=True)
        self.assertEqual(self.totals()["stop/truncated"], 1)
        self.metrics.reset(torch.tensor([0]))
        self.step(.5, .5)
        self.step(0, .3, failed=True, done=True)
        totals = self.totals()
        self.assertEqual(totals["stop/failures"], 1)
        self.assertEqual(totals["stop/truncated"], 0)
        self.metrics.reset(torch.tensor([0]))
        self.step(.5, .5)
        self.step(0, 0, 3)
        self.metrics.reset(torch.tensor([0]))
        self.assertEqual(self.totals()["stop/truncated"], 1)

    def test_unqualified_and_failed_restart_denominators(self):
        self.step(.5, 0)
        self.step(0, 0, 50)
        self.step(.3, .3)
        totals = self.totals()
        self.assertEqual(totals["stop/unqualified"], 1)
        self.assertEqual(totals["stop/completed"], 0)
        self.assertEqual(totals["restart/completed"], 0)
        self.step(0, 0, 50)
        self.step(.3, 0, 50)
        self.step(.4, 0)
        totals = self.totals()
        self.assertEqual(totals["stop/successes"], 1)
        self.assertEqual(totals["restart/failures"], 1)
        self.assertEqual(totals["restart/completed"], 1)

    def test_masks_band_boundaries_and_absent_samples(self):
        metrics = HLCMetrics(6, "cpu", .1, {})
        metrics.step(torch.tensor([0., .2, .5, 1., 1.5, .3]),
                     torch.tensor([0., .1, .4, 1.1, 1.6, 99.]), torch.zeros(6),
                     torch.zeros(6), torch.zeros(6), torch.tensor([True]*5 + [False]),
                     torch.zeros(6, dtype=torch.bool), torch.zeros(6, dtype=torch.bool))
        totals = {key: float(value) for key, value in metrics.drain().items()}
        summary = summarize_hlc_metrics(totals)
        for band, _, _ in metrics.SPEED_BANDS:
            self.assertEqual(summary["HLC_Speed/" + band + "/samples"], 1)
        self.assertAlmostEqual(summary["HLC_Speed/0_0p5/mae_m_s"], .1)
        self.assertEqual(summary["HLC_Events/zero_quiet_fraction"], 1)
        self.assertNotIn("HLC_Events/stop_success_rate", summary)
        empty = summarize_hlc_metrics({})
        self.assertNotIn("HLC_Speed/zero/mae_m_s", empty)

    def test_standing_is_not_a_small_positive_speed_restart(self):
        self.step(.5, .5)
        self.step(0, 0, 50)
        self.step(.05, 0, 50)
        self.step(.2, .2)
        totals = self.totals()
        self.assertEqual(totals["stop/successes"], 1)
        self.assertEqual(totals["restart/successes"], 0)
        self.assertEqual(totals["restart/failures"], 1)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
