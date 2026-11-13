# Adversarial Skill Embeddings

### F1 ASE LLC with the complete motion set

`HumanoidF1` uses 24 F1 reference clips from
`ase/data/motions/f1/dataset_f1.yaml`. The actor has 93 observations, the
privileged critic has 448, and the discriminator/encoder receive 890 motion
history features. ASE uses a 32-dimensional latent; no speed command is added.

The `joint_limits` mapping retains the default pose at action zero and uses
separate positive/negative scales with 10% joint-travel PD margin. This ASE
experiment starts from scratch; old models retain their saved control mapping.
PD gains, initialization, sampling weights, latent switching, and
discriminator/encoder reward weights (0.5/0.5) remain unchanged.

```bash
python ase/train_f1_cloud.py --algorithm ase --motion-file ase/data/motions/f1/dataset_f1.yaml --num_envs 4096 --max_iterations 3000 --seed 42
```

Seeds are configurable, with no fixed-seed or single-seed restriction. Logs
include policy/demo discriminator accuracy and logits, gradient penalty,
weighted discriminator/encoder rewards, initialization-specific episode
statistics, and knee/PD-target diagnostics. Startup logs and checkpoints record
the effective control configuration, reference files and hashes, and actual seed.

The shared preparation tool (outside this training repository) runs a small Gym
smoke with one ASE optimizer update, checkpoint roundtrip and camera-free replay:

```bash
python czy/f1_wbt/scripts/prepare_f1.py --ase --seed 42
```

Run that command from the parent project locally, or the flat project on 201.5.
On 201.5, activate the existing F1 environment and include
`/home/robot/czy/f1_ase/.deps` in `PYTHONPATH` to reuse its installed RL dependencies.
Smoke success checks the implementation, not learned motion quality.

Fixed-latent replay samples a different z per case; switching mode resamples
during each case. CSV records the case seed, actual latent and sample index.
`auto` preserves the historical case layout. For example:

```bash
python ase/replay_f1.py --engine gym --checkpoint MODEL --output RESULT --cases 8 --seed 42 --init-mode default --latent-mode fixed
python ase/replay_f1.py --engine gym --checkpoint MODEL --output RESULT_SWITCHING --cases 4 --seed 100 --init-mode default --latent-mode switching --no-video
```

`--duration` overrides replay length; `--no-video` produces CSV without cameras.
An unconditioned ASE latent can represent standing or backward motion, so one
random z cannot test the complete repertoire. Assess sustained motion, balance,
joint movement and behavior changes alongside reward statistics.

For a full-dataset diagnostic on 201.5 (Isaac Gym), run:

```bash
python ase/evaluate_f1_coverage.py --checkpoint MODEL --output RESULT --random-cases 64 --seed 20261113 --latent-seed 1000
```

This samples fixed random z from the default pose, then probes every reference
with its encoder z sequence from both default and matched reference states.
The latter probes are auxiliary retrieval tests; an unsuccessful encoded z does
not establish that a skill was never learned. The video shows policy output,
with clip, z source and initialization labels. No policy optimization occurs.
`isaac_coverage.mp4` and CSV, `coverage_summary.json`/CSV, and the compact
`coverage_features.npz` preserve per-case results, reference-transition distances
and phase coverage without overwriting the ordinary replay files.
Matching uses two consecutive heading-invariant 89-D AMP states, equal weights
for seven physical feature groups, and per-clip reference self-distance thresholds.
These bounded thresholds are diagnostic heuristics; nearest clip names and
reference-state coverage alone do not certify complete skills or transitions.
The first 0.5 seconds of every rollout are excluded from coverage support.
The report also isolates moving reference states (root speed above 0.2 m/s),
per-target encoder retrieval, and time-aligned distances. Recompute these from
saved features with the same command plus `--analyze-only`; no simulation runs.
Moving-phase support additionally requires compatible speed and direction, so
forward motion does not count as backward motion solely from a similar pose.

To extend selected encoder probes beyond their reference lengths, use one-based
dataset clip numbers and a fixed duration. After the reference ends, the last
encoded z is held; the reference does not loop. Both default and reference-state
initializations are recorded. A filename prefix preserves earlier artifacts:

```bash
python ase/evaluate_f1_coverage.py --checkpoint LLC.pt --output RESULT --random-cases 0 --clips 1 4 5 9 --duration 10 --prefix stop10_ --seed 20261113
```

The video labels the reference sequence and final-z hold separately. Tail speed
statistics cover the last two seconds; an early fall still ends that probe.
Keep the same prefix when using `--analyze-only` for these saved artifacts.

### F1 HLC speed control with a frozen ASE LLC

`HumanoidF1HLC` reuses the original HRL/PPO agent. Its actor receives the same
93 hardware observations as the LLC, followed by local target direction (2)
and target speed (1): 96 total. The separate HLC critic receives 448 privileged
features plus those 3 goal features. True linear velocity is used for rewards
and the critic, not appended to the actor. HLC outputs a continuous 32-D latent,
normalized to unit length, held for five LLC steps (HLC 10Hz / LLC 50Hz).

The initial task starts from the default pose at zero speed and samples target
speeds in [0.8,1.2]m/s every 5-8 seconds along fixed world +X. Its reward is
`exp(-4*(vx-target)^2 - 2*vy^2 - 2*(1-facing_x))`. HLC optimization combines
0.9 task reward and 0.1 frozen discriminator reward; the encoder is not trained.
The default timeout is 1500 LLC control steps, or 30 seconds.

The LLC loader restores only actor/discriminator weights and their normalization
statistics as frozen buffers, with no optimizer or training buffers. Its 93-D
input, 29-D previous joint action, PD gains, action mapping and physics remain
unchanged. HLC checkpoints are separate and bind the exact LLC SHA256, epoch and
dimensions; restoring/replaying with another LLC fails explicitly. Completed
episode lengths are logged in LLC steps/seconds; training frames count HLC
decisions. First terminal observations/reasons are latched within each interval,
and rewards after termination are excluded.

```bash
# Run from the parent project or the flat 201.5 project; one update, no skill test.
python czy/f1_wbt/scripts/prepare_f1.py --hlc --llc-checkpoint LLC.pt --seed 20261113
# Training entry; choose the run length when scheduling the actual experiment.
python ase/train_f1_cloud.py --algorithm hrl --llc-checkpoint LLC.pt --max_iterations 200 --num_envs 4096 --seed 20261113
# 30s sequence: 1.0 -> 0.8 -> 1.2 -> 1.0m/s; add --no-video for CSV only.
python ase/replay_f1_hlc.py --checkpoint HLC.pth --llc-checkpoint LLC.pt --output RESULT
```

Cloud launches can additionally supply `--llc-checkpoint-url` and `--llc-sha256`.
The entry downloads the dependency to the specified external cache path, verifies
its hash before use, and excludes the signed URL from its own logs/checkpoints.

Replay writes `isaac_hlc.mp4` and a 50Hz `isaac_hlc.csv` with command, world/body
velocity, pose, z, task reward and termination. Smoke success verifies the
engineering path and LLC immutability; it does not establish speed tracking,
stopping, running, steering or backward locomotion quality.

**This codebase is now deprecated.** Checkout [MimicKit](https://github.com/xbpeng/MimicKit/) for much more up-to-date and easy-to-use implementations of our methods.

Code accompanying the paper:
"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" \
(https://xbpeng.github.io/projects/ASE/index.html) \
![Skills](images/ase_teaser.png)


### Installation

Download Isaac Gym from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:

```
pip install -r requirements.txt
```


### ASE

#### Pre-Training

First, an ASE model can be trained to imitate a dataset of motions clips using the following command:
```
python ase/run.py --task HumanoidAMPGetup --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --headless
```
`--motion_file` can be used to specify a dataset of motion clips that the model should imitate. 
The task `HumanoidAMPGetup` will train a model to imitate a dataset of motion clips and get up after falling.
Over the course of training, the latest checkpoint `Humanoid.pth` will be regularly saved to `output/`,
along with a Tensorboard log. `--headless` is used to disable visualizations. If you want to view the
simulation, simply remove this flag. To test a trained model, use the following command:
```
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint [path_to_ase_checkpoint]
```
You can also test the robustness of the model with `--task HumanoidPerturb`, which will throw projectiles at the character.

&nbsp;

#### Task-Training

After the ASE low-level controller has been trained, it can be used to train task-specific high-level controllers.
The following command will use a pre-trained ASE model to perform a target heading task:
```
python ase/run.py --task HumanoidHeading --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint [path_to_llc_checkpoint] --headless
```
`--llc_checkpoint` specifies the checkpoint to use for the low-level controller. A pre-trained ASE low-level
controller is available in `ase/data/models/ase_llc_reallusion_sword_shield.pth`.
`--task` specifies the task that the character should perform, and `--cfg_env` specifies the environment
configurations for that task. The built-in tasks and their respective config files are:
```
HumanoidReach: ase/data/cfg/humanoid_sword_shield_reach.yaml
HumanoidHeading: ase/data/cfg/humanoid_sword_shield_heading.yaml
HumanoidLocation: ase/data/cfg/humanoid_sword_shield_location.yaml
HumanoidStrike: ase/data/cfg/humanoid_sword_shield_strike.yaml
```
To test a trained model, use the following command:
```
python ase/run.py --test --task HumanoidHeading --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint [path_to_llc_checkpoint] --checkpoint [path_to_hlc_checkpoint]
```


&nbsp;

&nbsp;

#### Pre-Trained Models

Pre-trained models are provided in `ase/data/models/`. To run a pre-trained ASE low-level controller,
use the following command:
```
python ase/run.py --test --task HumanoidAMPGetup --num_envs 16 --cfg_env ase/data/cfg/humanoid_ase_sword_shield_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/dataset_reallusion_sword_shield.yaml --checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth
```

Pre-trained models for the different tasks can be run using the following commands:

Heading:
```
python ase/run.py --test --task HumanoidHeading --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield_heading.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_heading_reallusion_sword_shield.pth
```

Reach:
```
python ase/run.py --test --task HumanoidReach --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield_reach.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_reach_reallusion_sword_shield.pth
```

Location:
```
python ase/run.py --test --task HumanoidLocation --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield_location.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_location_reallusion_sword_shield.pth
```

Strike:
```
python ase/run.py --test --task HumanoidStrike --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield_strike.yaml --cfg_train ase/data/cfg/train/rlg/hrl_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Idle_Ready_Motion.npy --llc_checkpoint ase/data/models/ase_llc_reallusion_sword_shield.pth --checkpoint ase/data/models/ase_hlc_strike_reallusion_sword_shield.pth
```

&nbsp;

&nbsp;

### AMP

We also provide an implementation of Adversarial Motion Priors (https://xbpeng.github.io/projects/AMP/index.html).
A model can be trained to imitate a given reference motion using the following command:
```
python ase/run.py --task HumanoidAMP --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --headless
```
The trained model can then be tested with:
```
python ase/run.py --test --task HumanoidAMP --num_envs 16 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy --checkpoint [path_to_amp_checkpoint]
```

&nbsp;

&nbsp;

### Motion Data

Motion clips are located in `ase/data/motions/`. Individual motion clips are stored as `.npy` files. Motion datasets are specified by `.yaml` files, which contains a list of motion clips to be included in the dataset. Motion clips can be visualized with the following command:
```
python ase/run.py --test --task HumanoidViewMotion --num_envs 2 --cfg_env ase/data/cfg/humanoid_sword_shield.yaml --cfg_train ase/data/cfg/train/rlg/amp_humanoid.yaml --motion_file ase/data/motions/reallusion_sword_shield/RL_Avatar_Atk_2xCombo01_Motion.npy
```
`--motion_file` can be used to visualize a single motion clip `.npy` or a motion dataset `.yaml`.


This motion data is provided courtesy of Reallusion, strictly for noncommercial use. The original motion data is available at:

https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-stunts

https://actorcore.reallusion.com/motion/pack/studio-mocap-sword-and-shield-moves


If you want to retarget new motion clips to the character, you can take a look at an example retargeting script in `ase/poselib/retarget_motion.py`.
