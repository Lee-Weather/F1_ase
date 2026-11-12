# Adversarial Skill Embeddings

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

### F1 velocity-conditioned AMP/ASE

`HumanoidF1Velocity` adds forward speed commands. The legacy ASE path retains
the 32-dimensional latent and encoder, with one latent held per episode.
Actor observations are 93 proprioceptive values plus three commands (96 total);
the privileged critic has 451 inputs. The actor does not receive simulated base
linear velocity or reference frames. Lateral/yaw commands are reserved at zero.

The environment extends `humanoid_f1.yaml`, retaining the asset, PD gains and
joint-limit action mapping. Version 2 limits policy PD targets for both hip yaw
joints to ±40 degrees, without changing physical limits or adding PD margin to
these two channels. The same five NPZs train the discriminator. Foot heading,
hip yaw, stance sole tilt and out-of-range PD targets have soft costs. Forward
tracking uses `1 / (1 + ((vx-command)/0.3)^2)`; lateral/yaw/tilt deviations are
costs. Quiet standing also favors the middle-half median posture of A1.

Task/discriminator/encoder weights remain 1.0/0.3/0.05. Encoder reward and loss
use the same transition mask: valid history × command gate × forward-motion
gate. Motion is measured from 0.2 seconds of root displacement, with a 0.05–0.15
m/s gate; it is used only for rewards/losses/diagnostics. Commands retain
acceleration/deceleration limits of 0.3/0.5 m/s².

Curriculum A starts 70% of episodes from B3 and 30% from the default posture;
default starts wait 0.3–0.5 seconds, then request 0.4–0.6 m/s continuously.
Reference starts match their initial forward speed before ramping to a target.
Stage B uses 70% default starts, all five reference initializations and
stand/walk/restart at 0.3–0.6 m/s. Stage C expands walking to 0.2–0.8 m/s.
Walking/standing segments last 6–10/3–5 seconds; a mid-walk retarget waits until
the current command reaches its target and stays there for two seconds.
Stages advance explicitly after replay review, never automatically by epoch.

After simulation validation and training authorization, run from the repository root:

```bash
python ase/train_f1_cloud.py --task HumanoidF1Velocity --algorithm ase --num_envs 4096 --max_iterations 500 --velocity-stage A --seed 42
```

The entry selects the velocity environment, training configuration and motion list.
`--cfg-env`, `--cfg-train` and `--motion-file` allow explicit overrides. Old F1 AMP/ASE
defaults remain available under `--task HumanoidF1`. Velocity checkpoints contain
the observation schema, command settings and control metadata; they cannot be
resumed as the old 93-input task. The default pilot budget is 500 iterations.
The checkpoint also stores curriculum stage separately from the immutable policy
schema. Resume restores that stage; `--velocity-stage B` or `C` explicitly
overrides it. Start v2 from scratch, rather than resuming a failed v1 policy.
Replay accepts v1/v2 metadata and restores the saved control window; a missing
window means the original mapping, so legacy policies do not inherit ±40°.

Velocity AMP uses `--algorithm amp` with the same task. The entry selects the
small `humanoid_f1_velocity_amp.yaml` overlay and `amp_f1_velocity.yaml`.
It uses actor/critic/discriminator layers [1024, 1024, 512], task/discriminator
weights 1.0/0.3, and no latent, style network, encoder, encoder reward or mask.
The AMP actor takes only the 96 observations including the speed command.
AMP starts in stage B (stand/walk/stop/restart, 0.3–0.6 m/s); stage C extends
the range to 0.8 m/s after review. Stage A remains an explicit walking-only
option. There is no automatic stage advancement.

Executed AMP knee PD targets are clamped to their physical joint limits.
The raw, unclamped requests remain in the joint-limit cost, averaging knees
separately to avoid dilution across 29 joints. Default joint positions, other
control channels, physical limits, KP/KD and armature retain their old values.
Schema 3 records algorithm AMP, latent_dim=0 and a separate task_reward_version;
old ASE schemas 1/2 and old 93-input AMP checkpoints retain saved controls.
Start AMP from scratch; ASE optimizer/network weights cannot be resumed into AMP.
Console logs include discriminator loss/reward, policy/demo accuracy, logits
and gradient penalty. Accuracy treats policy samples as negative and demos as
positive; these are classifier diagnostics, not walking acceptance metrics.

After training authorization (this command does not resume ASE):

```bash
python ase/train_f1_cloud.py --task HumanoidF1Velocity --algorithm amp --num_envs 4096 --max_iterations 500 --velocity-stage B --seed 42
```

Formal acceptance uses one seed (42), default posture and zero initial velocity,
with one continuous 45-second sequence and no resets:
0–5 s: 0; 5–13: 0.3; 13–21: 0.6; 21–29: 0.8; 29–35: 0.3; 35–45: 0 m/s.
A short walking-only check cannot replace this acceptance.

```bash
python ase/replay_f1.py --engine gym --checkpoint /path/to/model.pt --output /path/to/czy/data/exp_f1_6 --command-profile walk-stop --duration 45 --cases 1 --seed 42
```

Summaries report steady speed MAE, stop time and final-five-second path length.
Stop requires speed <0.05 m/s, **maximum** foot speed <0.1 m/s and leg joint RMS
speed <0.3 rad/s continuously for 0.5 s, within 3 s of the zero target. The final
five seconds must remain quiet with path <=0.1 m; steady MAE must be <=0.12 m/s.
A finished sequence is `sequence_complete`, independently of numeric targets
and human video review. Replay defaults to one case and AMP never samples z.

Interactive input is optional and uses [pygame joystick](https://www.pygame.org/docs/ref/joystick.html)
(`pygame>=2` is needed only for this mode):

```bash
python ase/replay_f1.py --engine gym --checkpoint /path/to/model.pt --output /path/to/czy/data/exp_f1_5 --command-profile joystick --viewer --no-video --cases 1 --duration 60
```

The default axis is 1, negative direction for forward; `--axis`, `--axis-sign`,
`--joystick` and `--stop-button` configure the device. Button 0 requests zero speed;
a disconnected device also requests zero. The task applies the same command ramp
used during training. No remote connection or automatic package installation is
performed by replay.

CPU functional checks require the existing torch/NumPy/PyYAML/rl-games packages,
and do not require Isaac Gym, a server, or optimizer updates:

```bash
python -m unittest discover -s ase/tests -v
```

These checks cover command/reward timing with Gym I/O replaced, the actual ASE
network and portable inference, masks and shuffled minibatches. They do not
validate physical locomotion; that remains a separate Isaac Gym run.

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
