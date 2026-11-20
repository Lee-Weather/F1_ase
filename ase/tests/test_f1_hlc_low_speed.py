"""Low-speed curriculum boundaries, reward incentives, and replay parity."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.f1_hlc import speed_task_reward
from utils.f1_hlc_curriculum import command_stage
from replay_f1_hlc import numpy_task_reward


class LowSpeedTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(speedTrackingScale=4., lateralTrackingScale=2., headingTrackingScale=2.,
                        standFootSpeedScale=2., standLegVelocityScale=.05,
                        movingSpeedSigmaMin=.1, movingSpeedSigmaFraction=.35,
                        standingSpeedSigma=.05, requireForwardProgress=True)

    def reward(self, target, vx):
        target = torch.tensor(target, dtype=torch.float32)
        v = torch.zeros(len(target), 3);v[:, 0] = torch.tensor(vx)
        return speed_task_reward(v, torch.ones(len(target)), target, torch.zeros_like(target), torch.zeros_like(target), self.cfg)

    def test_standing_cannot_earn_positive_command_reward(self):
        self.assertTrue(torch.equal(self.reward([.1,.2,.3,.5,1.], [0.]*5), torch.zeros(5)))
        self.assertTrue(torch.equal(self.reward([.1,.2,.3,.5,1.], [.1,.2,.3,.5,1.]), torch.ones(5)))
        self.assertLess(float(self.reward([.1], [.2])[0]), 1.)
        self.assertEqual(float(self.reward([.1], [-.1])[0]), 0.)

    def test_zero_command_penalizes_drift(self):
        self.assertEqual(float(self.reward([0.], [0.])[0]), 1.)
        self.assertAlmostEqual(float(self.reward([0.], [.1])[0]), math.exp(-4), places=6)
        self.assertAlmostEqual(numpy_task_reward(np.array([0.,.1,0.]),1.,0.,0.,0.,self.cfg),math.exp(-4),places=6)

    def test_numpy_torch_reward_parity_and_legacy(self):
        rng = np.random.RandomState(20261120)
        targets = np.array([0.,.1,.2,.3,.5,1.]*20, dtype=np.float32)
        velocities = rng.normal(0, .3, (len(targets),3)).astype(np.float32)
        facing = rng.uniform(-1,1,len(targets)).astype(np.float32)
        foot = rng.uniform(0,1,len(targets)).astype(np.float32)
        legs = rng.uniform(0,1,len(targets)).astype(np.float32)
        for cfg in [self.cfg, {k:v for k,v in self.cfg.items() if k not in ('movingSpeedSigmaMin','movingSpeedSigmaFraction','standingSpeedSigma','requireForwardProgress')}]:
            actual = speed_task_reward(torch.from_numpy(velocities),torch.from_numpy(facing),torch.from_numpy(targets),torch.from_numpy(foot),torch.from_numpy(legs),cfg).numpy()
            expected = [numpy_task_reward(v,f,t,p,l,cfg) for v,f,t,p,l in zip(velocities,facing,targets,foot,legs)]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
        self.assertAlmostEqual(numpy_task_reward(np.zeros(3),1.,.5,0.,0.,cfg),math.exp(-1),places=7)

    def test_curriculum_boundaries_and_final_stage(self):
        path = Path(__file__).resolve().parents[1]/'data/cfg/humanoid_f1_hlc_0_1.yaml'
        cfg = yaml.safe_load(path.read_text())['env']['hlc']
        for epoch, stage in [(0,1),(200,1),(201,2),(500,2),(501,3),(1000,3),(6000,3)]:
            index, spec = command_stage(cfg,epoch)
            self.assertEqual(index,stage)
            self.assertLessEqual(spec['speedRange'][1],1.)
        self.assertEqual(command_stage({'speedRange':[0,2]},100)[0],0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
