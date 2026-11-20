# F1 行走运动控制：ASE LLC与HLC速度控制

面向F1 29自由度人形机器人的仿真运动控制项目。基于ASE学习可复用的底层技能控制器（LLC），再冻结LLC、训练高层控制器（HLC），实现目标速度下的起步、行走、调速与停止，并在Isaac Gym和MuJoCo中验证。

当前任务沿固定世界+X方向控制速度，训练指令范围为0～2m/s。当前工作范围是仿真训练与回放；真机部署、实体手柄、任意方向转向和稳定跑步尚未验收。

## 当前进展

截至2026-11-20，**exp_f1_8.2的2850轮HLC作为当前调速主基线，850轮保留为停稳对照；8.3/8.4分别对比7.1与7.2-3000 LLC，8.5准备使用7.2-6000 LLC。** 8.2～8.4均未完成原定3000轮，8.5使用6000轮LLC进行1800轮HLC对照。Gym调速有效，完整双引擎站走停尚未通过。

| 实验 | 模型与用途 | 已验证结果 |
|---|---|---|
| exp_f1_7 | 24段参考训练ASE LLC，3000轮 | 部分latent可自主起步和持续行走；完整参考技能覆盖及走→停未通过 |
| exp_f1_7.1 | 36段参考、增强停步数据的ASE LLC，3000轮 | Gym行走/静站可用；12组所测走→站立latent切换均未停稳 |
| exp_f1_7.2 | 原24段＋8段慢走/过渡参考的ASE LLC，3000轮 | 88组Gym诊断部分达标：出现较慢自主步态，但0.2～0.5m/s稳态覆盖待验证，有效固定z走停0/7 |
| exp_f1_7.2-6000 | 同一32段LLC由3000轮续训至6000轮 | 92组Gym诊断部分达标：固定z持续前进14/64，出现末5秒约0.4m/s步态；重筛组合有效走停1/4、再起步0/1 |
| exp_f1_8.2 | 冻结原7 LLC；最新保存HLC为2850轮，末日志2854轮 | Gym完整67.5秒无跌倒，速度MAE 0.06895m/s；最后停步末5秒路径10.24cm，略超10cm阈值；本机MuJoCo在52.7秒再起步时失稳 |
| exp_f1_8.3 | 冻结7.1 LLC；最新保存HLC为1700轮，末日志1704轮 | Gym完整67.5秒无跌倒，速度MAE 0.07043m/s；首次1→0未停稳，末次0.5→0通过本次停稳阈值；本机MuJoCo在初始零速段4.6秒失稳 |
| exp_f1_8.4 | 冻结7.2的3000轮LLC；最新保存HLC为1800轮，末日志1829轮 | Gym完整67.5秒，速度MAE 0.07129m/s，两次走停阈值未通过；本机MuJoCo首次起步9.54秒失稳 |
| exp_f1_8.5 | 冻结7.2的6000轮LLC；从零训练1800轮HLC | 保持8.4高层任务及优化参数，重点与8.4的1800轮模型比较；启动状态和结果见工作区实验记录 |

8.2的850轮与2850轮在相同协议下，Gym速度MAE由0.08189降至0.06895m/s，下降约15.8%；但最后停步路径由3.56cm增至10.24cm，停稳能力没有随轮数全面改善。两模型是450轮后分开的训练分支。8.2与8.3的HLC训练轮数也不同，不能仅凭这些单seed回放判定新增参考数据或7.1 LLC本身更差。

## 控制架构

```mermaid
flowchart LR
    C[目标方向与速度] --> H[HLC：10Hz]
    S[机器人状态] --> H
    H --> Z[32维单位latent z]
    Z --> L[冻结的ASE LLC：50Hz]
    S --> L
    L --> A[29维关节动作]
    A --> P[关节目标映射与PD控制]
    P --> R[F1仿真]
    R --> S
```

| 模块 | 接口与训练方式 |
|---|---|
| ASE LLC | actor为93维状态＋32维z，输出29维关节动作；critic为448维；AMP判别器/encoder使用10帧×89维历史，共890维 |
| HLC | actor为93维状态＋2维局部目标方向＋1维目标速度，共96维；critic为451维；输出归一化的32维z |
| 控制频率 | LLC 50Hz；每5个LLC步更新一次z，HLC 10Hz |
| 冻结依赖 | HLC加载LLC actor、判别器及其归一化统计，保持冻结；HLC使用自己的actor/critic和优化器 |

当前HLC actor没有显式的根部实际线速度或速度误差输入；真实线速度用于critic和奖励。actor仍通过角速度、重力投影、关节状态及上一动作获得反馈。跨引擎执行器/接触响应差异与减速失稳的具体因果关系尚未定位。

动作采用`joint_limits`分方向映射：零动作对应默认站姿，正负方向分别覆盖关节限位并保留10%行程PD余量。旧checkpoint按其保存的控制配置回放。KP/KD、armature等由环境配置和模型元数据追溯。

HLC checkpoint记录所依赖LLC的SHA256、轮次和网络维度。回放或恢复必须使用对应LLC；从8.2切换到7.1 LLC需要从零训练新的HLC，不能直接替换依赖。

## 目录与配置

以下路径相对本训练仓库根目录。

| 路径 | 内容 |
|---|---|
| [ase/train_f1_cloud.py](ase/train_f1_cloud.py) | F1 ASE/AMP/HRL统一训练入口、依赖准备、恢复训练与日志 |
| [ase/env/tasks/humanoid_f1.py](ase/env/tasks/humanoid_f1.py) | F1模型、状态、动作映射与终止条件 |
| [ase/env/tasks/humanoid_f1_hlc.py](ase/env/tasks/humanoid_f1_hlc.py) | HLC速度指令与站走停任务奖励 |
| [ase/utils/f1_hlc.py](ase/utils/f1_hlc.py) | 冻结LLC、HLC网络和高低层控制间隔 |
| [ase/data/cfg/humanoid_f1.yaml](ase/data/cfg/humanoid_f1.yaml) | LLC环境、关节默认姿态、PD及物理参数 |
| [ase/data/cfg/humanoid_f1_hlc_0_2.yaml](ase/data/cfg/humanoid_f1_hlc_0_2.yaml) | 当前0～2m/s、20%精确零速的HLC环境 |
| [ase/data/cfg/train/rlg/ase_f1.yaml](ase/data/cfg/train/rlg/ase_f1.yaml) | ASE LLC训练参数 |
| [ase/data/cfg/train/rlg/hrl_f1.yaml](ase/data/cfg/train/rlg/hrl_f1.yaml) | HLC PPO训练参数 |
| [ase/replay_f1.py](ase/replay_f1.py) | LLC双引擎策略回放 |
| [ase/replay_f1_hlc.py](ase/replay_f1_hlc.py) | HLC双引擎调速/站走停回放及NumPy权重导出 |
| [ase/evaluate_f1_coverage.py](ase/evaluate_f1_coverage.py) | 随机z、参考编码z及参考技能覆盖诊断 |
| [ase/evaluate_f1_transitions.py](ase/evaluate_f1_transitions.py) | 连续走→站立z切换诊断 |

实验记录与模型产物在父工作区的`czy/`中维护，不随训练Git仓库分发：

- `../czy/exp1/exp_llc.md`：LLC与早期直接控制实验1～7.1。
- `../czy/exp1/exp_hlc.md`：HLC实验8～8.3，含850/2850轮和8.2/8.3对照。
- `../czy/exp1/exp1.md`：实验总索引。
- `../czy/plan/f1_phase1.md`：项目计划、准备与历史方案。
- `../czy/data/<实验编号>/`：已归档checkpoint、MP4、CSV及必要推理权重。

201.5服务器使用平铺布局，项目根目录为`/home/robot/czy/f1_ase`，其中源码在`ase/`、记录与产物在`czy/`。下方`../czy/`模型路径适用于本机父工作区；服务器平铺布局应改为`czy/`或绝对路径。

## 环境准备

训练与Gym回放使用Linux、NVIDIA GPU和Isaac Gym Preview 4。201.5已验证环境为Python 3.8.20、PyTorch 1.12.1+cu113；云端训练也使用过自带PyTorch 2.4.1的镜像。复用匹配Isaac Gym的现有环境，保留镜像自带的PyTorch/CUDA。

训练入口的`--install-deps`按`PACKAGES`安装配套依赖，包括NumPy 1.23.5、rl-games 1.1.4、gym 0.23.1、ray 1.13.0等；依赖目录通过`F1_DEPS_DIR`设置，默认位于临时目录的`f1_ase_deps`。根目录`requirements.txt`保留了上游旧版本列表，当前F1训练以入口的依赖配置为准。

201.5复用已有依赖的示例（在项目根目录执行）：

```bash
conda activate F1
export F1_DEPS_DIR=/home/robot/czy/f1_ase/.deps
export PYTHONPATH="$F1_DEPS_DIR:ase:${PYTHONPATH:-}"
```

MuJoCo回放可独立于Isaac Gym运行。本机已验证MuJoCo 3.15.0、NumPy 1.26.4；录制还需要回放脚本使用的Pillow、imageio-ffmpeg等依赖和可用渲染环境。本机PyTorch不可用时使用下述NumPy推理路径。

## 参考动作与LLC训练

| 参考清单 | 规模与用途 |
|---|---|
| [dataset_f1.yaml](ase/data/motions/f1/dataset_f1.yaml) | 原24段、5992帧；exp_f1_7及8.2的LLC基座 |
| [dataset_f1_stop_data.yaml](ase/data/motions/f1/dataset_f1_stop_data.yaml) | 36段、10199帧；exp_f1_7.1及8.3的LLC基座 |

F1使用50Hz tracking NPZ参考数据。36段版本的逐条动作和权重见[README_stop_data.md](ase/data/motions/f1/README_stop_data.md)：原20条其他运动合计60%、新增11条其他运动10%、4条停步25%、A1站立5%。数据含行走、跑步、站立及过渡，数据入选不代表策略已掌握全部技能。

下列命令从训练仓库根目录执行，首次准备依赖时加入`--install-deps`。seed仅为示例，可按实验选择数量和值，并记录实际使用值。

```bash
# 原24段ASE LLC，从零训练。
python ase/train_f1_cloud.py --algorithm ase --motion-file ase/data/motions/f1/dataset_f1.yaml --num_envs 4096 --max_iterations 3000 --seed 42 --json-logs

# 36段、增强停步数据的ASE LLC，从零训练。
python ase/train_f1_cloud.py --algorithm ase --motion-file ase/data/motions/f1/dataset_f1_stop_data.yaml --num_envs 4096 --max_iterations 3000 --seed 42 --json-logs
```

LLC本身不接收目标速度，z代表技能输入。固定z表示在一段回放中保持同一个latent；不同case可取不同z。随机z可能产生站立、前进或其他行为，不能用单个z评价完整动作集。

## HLC速度控制训练

当前HLC从默认站姿、零初速开始。每5～8秒更新目标速度：20%为精确0，其余80%在[0,2)m/s均匀采样；训练回合上限为1500个LLC控制步，即30秒。

任务奖励为：

```text
exp(-4*(vx-target_speed)^2 - 2*vy^2 - 2*(1-facing_x)
    - I(target_speed=0)*(2*foot_speed_sq + 0.05*leg_velocity_sq))
```

`foot_speed_sq`为双脚线速度平方范数的均值，`leg_velocity_sq`为腿部关节速度平方均值。HLC优化奖励为0.9×任务奖励＋0.1×冻结判别器奖励；LLC不参与优化。

```bash
# 8.2配置：冻结原24段LLC，从零训练HLC。
python ase/train_f1_cloud.py --algorithm hrl --cfg-env ase/data/cfg/humanoid_f1_hlc_0_2.yaml --motion-file ase/data/motions/f1/dataset_f1.yaml --llc-checkpoint ../czy/data/exp_f1_7/model_3000.pt --num_envs 4096 --max_iterations 3000 --seed 20261116 --json-logs

# 8.3配置：冻结36段7.1 LLC，从零训练新的HLC。
python ase/train_f1_cloud.py --algorithm hrl --cfg-env ase/data/cfg/humanoid_f1_hlc_0_2.yaml --motion-file ase/data/motions/f1/dataset_f1_stop_data.yaml --llc-checkpoint ../czy/data/exp_f1_7.1/model_3000.pt --num_envs 4096 --max_iterations 3000 --seed 20261116 --json-logs
```

这些命令用于复现配置，实际运行会创建新的训练过程。`humanoid_f1_hlc.yaml`保留0.6～1.6m/s历史任务；当前0～2m/s实验须显式选择`humanoid_f1_hlc_0_2.yaml`。

### exp_f1_8.4：基于7.2 LLC的低速与站走停对照

8.4保持8.2的命令分布、奖励、网络和物理参数，冻结7.2的3000轮LLC，从零训练HLC；实际训练和回放结果见上表。32段数据清单为`dataset_f1_7_2.yaml`，不是7.1的36段或撤回的7.3方案；模型SHA256绑定防止混用依赖。

```bash
python ase/train_f1_cloud.py --algorithm hrl --cfg-env ase/data/cfg/humanoid_f1_hlc_0_2.yaml --motion-file ase/data/motions/f1/dataset_f1_7_2.yaml --llc-checkpoint ../czy/data/exp_f1_7.2/model_3000.pt --llc-sha256 5f0f31d3a1d998c9195d391002ca8f43bc69ccacd301a0b49b348c32fec5e88e --num_envs 4096 --max_iterations 3000 --archive-iterations 850 1700 2850 3000 --seed 20261116 --json-logs
```

`--archive-iterations`只在指定轮数增加命名checkpoint，保留原每50轮覆盖式最新保存；例如`F1_HLC_00000850.pth`用于同轮数对照。它不改变优化步数或奖励，也不恢复旧HLC。

### exp_f1_8.5：冻结7.2的6000轮LLC

8.5替换为`exp_f1_7.2/model_6000.pt`，固定SHA256为`1d55ac95f1603eb16e069a5aeaf5a47d956a928e3fb7e70e61c3d8fcc7ab12ed`。数据清单、动作映射、PD、HLC网络、0～2m/s/20%零速命令、奖励和PPO设置与8.4一致。HLC从零训练，不能加载8.4 HLC或将其依赖直接改为6000轮LLC；6000轮后的latent语义和归一化已变化。

```bash
python ase/train_f1_cloud.py --algorithm hrl --cfg-env ase/data/cfg/humanoid_f1_hlc_0_2.yaml --motion-file ase/data/motions/f1/dataset_f1_7_2.yaml --llc-checkpoint ../czy/data/exp_f1_7.2/model_6000.pt --llc-sha256 1d55ac95f1603eb16e069a5aeaf5a47d956a928e3fb7e70e61c3d8fcc7ab12ed --num_envs 4096 --max_iterations 1800 --archive-iterations 850 1700 1800 --seed 20261116 --json-logs
```

用户指定8.5从零训练1800轮HLC，保留850/1700/1800轮命名模型，与8.4实际保存的1800轮做同预算回放。现有入口无需增加新算法或独立启动脚本；Flux入口仍是`gm-run F1_ase/ase/train_f1_cloud.py`，hparams为`F1_ase/ase/data/cfg/train/rlg/hrl_f1.yaml`，云端缓存模型使用独立的6000轮文件名。正式训练/完整技能验收开始前通知，账号与云任务在正式启动时另行确定。

续训时增加`--resume-checkpoint HLC.pt`，保留匹配的LLC、环境和参考清单。它恢复HLC模型、归一化、Adam和轮次等训练状态；`--max_iterations`表示总目标轮数，例如450轮恢复到总3000轮。仿真环境重新初始化，不表示恢复原容器的连续物理状态。

云端入口可通过`gm-run`调用本脚本，并传入`--llc-checkpoint-url`/`--llc-sha256`以及对应的resume参数下载和核验依赖；下载链接只用于运行时，不写入项目文档或源码。

## 回放与验收

Gym默认在201.5录制，MuJoCo默认在本机录制。HLC的配对协议为：

```text
目标速度(m/s)：0 → 0.5 → 1 → 1.5 → 2 → 1 → 0 → 0.5 → 0
每段7.5秒，共67.5秒；默认站姿零初速，中间不重置，失稳则提前结束。
```

使用当前8.2主基线的示例（每次使用新的输出目录，避免覆盖既有回放）：

```bash
python ase/replay_f1_hlc.py --engine gym --checkpoint ../czy/data/exp_f1_8.2/model_2850_resume.pt --llc-checkpoint ../czy/data/exp_f1_7/model_3000.pt --output review/hlc_2850_gym --protocol stop-start --hold-seconds 7.5 --seed 20261116

python ase/replay_f1_hlc.py --engine mujoco --checkpoint ../czy/data/exp_f1_8.2/model_2850_resume.pt --llc-checkpoint ../czy/data/exp_f1_7/model_3000.pt --output review/hlc_2850_mujoco --protocol stop-start --hold-seconds 7.5 --seed 20261116
```

Gym输出`isaac_hlc.mp4`和`isaac_hlc.csv`；MuJoCo输出`mujoco_hlc.mp4`和`mujoco_hlc.csv`。CSV按50Hz记录指令、世界/机身速度、姿态、z、脚/腿RMS、任务奖励及终止原因。`--no-video`仅导出CSV；`--speeds 1,0.5,0`可指定额外减速序列。

8.4增加`--protocol low-speed`（0→0.2→0.3→0.5→0.3→0.2→0→0.3→0，默认每段10秒，共90秒）和`--protocol decel-restart`（1→0.5→0.2→0→0.2→0.5→1，默认每段10秒，共70秒）。两引擎使用相同协议，`--hold-seconds`可覆盖时长；原tracking和stop-start默认每段7.5秒不变。

`HLC_REPLAY_SEGMENTS`按指令段统计，稳态速度误差排除每段首2秒。零速验收同时要求：

- 水平速度<0.05m/s、脚RMS<0.1m/s、腿部关节速度RMS<0.3rad/s，且无跌倒。
- 3秒内达到连续安静0.5秒；日志的`settle_s`是该连续窗口的完成时间。
- 停止段末5秒路径≤0.1m；提前终止的短段不能当成完整末5秒验收。

低前向速度、初始静站或Gym无跌倒，都不能单独代替走→停、再次起步和MuJoCo迁移验收。

### 本机无可用PyTorch时

在可用PyTorch的主机导出HLC及其匹配LLC的确定性actor，再把NPZ与原checkpoint一起复制到回放主机：

```bash
PYTHONPATH=ase python -c "from replay_f1_hlc import export_numpy_replay; export_numpy_replay('HLC.pt', 'LLC.pt', 'hlc_actor.npz', 'llc_actor.npz')"

python ase/replay_f1_hlc.py --engine mujoco --checkpoint HLC.pt --llc-checkpoint LLC.pt --policy-npz hlc_actor.npz --llc-policy-npz llc_actor.npz --output review/hlc_numpy --protocol stop-start --hold-seconds 7.5 --seed 20261116
```

NumPy路径保留HLC tanh、单位z、LLC归一化及动作裁剪，并校验两个源checkpoint哈希和LLC绑定关系。既有128组输入的Torch/NumPy输出对照误差小于1e-5。8.3必须使用7.1 LLC对应的NPZ，不能复用原7 LLC权重。Windows本机可将`OPENBLAS_NUM_THREADS`、`OMP_NUM_THREADS`、`MKL_NUM_THREADS`设为1，避免小批量推理的线程开销。

### LLC回放与技能诊断

```bash
# 默认站姿、不同case使用不同固定z。
python ase/replay_f1.py --engine gym --checkpoint LLC.pt --output review/llc_fixed --cases 8 --seed 20261116 --init-mode default --latent-mode fixed

# 36段LLC：随机z与每段参考编码z的双起步探针。
python ase/evaluate_f1_coverage.py --checkpoint LLC.pt --dataset ase/data/motions/f1/dataset_f1_stop_data.yaml --output review/llc_coverage --random-cases 64 --seed 20261116 --latent-seed 1000

# 已知走路/站立z的连续切换；seed对需使用对应LLC的诊断结果。
python ase/evaluate_f1_transitions.py --checkpoint LLC.pt --dataset ase/data/motions/f1/dataset_f1_stop_data.yaml --output review/llc_transitions --walk-seeds 1002 1026 --stand-seeds 1000 1003 --blend-seconds 0 0.5 --walk-seconds 3 --duration 10 --seed 20261116 --prefix switchstop_
```

诊断脚本导出视频、逐步CSV和汇总。`--dataset`应与被测LLC的数据版本对应；参考编码z探针用于辅助检索，失败不能独立证明技能不存在。覆盖统计使用距离阈值等启发式规则，不能直接等同完整技能验收。LLC回放的`--motion-file`用于起步状态，不在闭环中持续输入参考姿态。

父工作区的`czy/f1_wbt/scripts/prepare_f1.py`提供必要的ASE/HLC短冒烟；需从父工作区或201.5平铺根目录运行。冒烟验证数据、优化更新、冻结依赖与保存/恢复链路，不能代替训练后的运动能力验收。

## 训练日志

优先关注速度误差、横向运动、实际回合秒数、终止原因，以及零速时脚/腿是否安静。

| 日志 | 含义 |
|---|---|
| `Metrics/motion/speed_error_abs_m_s` | 前向速度与目标速度的绝对误差 |
| `Metrics/motion/lateral_speed_abs_m_s` | 横向速度绝对值 |
| `Mean episode duration (s)`、`Episode_Termination/*` | 回合时长与终止原因；原因计数可能重叠 |
| `Metrics/motion/stop_command_fraction` | 全部样本中零速指令的比例 |
| `Metrics/motion/stop_quiet_fraction` | 全部样本中“零速且即时安静”的比例；除以上一项才是零速条件下即时安静比例，不是连续停稳成功率 |
| `Metrics/motion/foot_speed_rms_m_s`、`leg_velocity_rms_rad_s` | 脚和腿的运动程度，结合零速段判断原地踏步 |
| `Reward/weighted_task`、`Reward/weighted_discriminator` | 实际优化奖励的加权组成 |
| `Discriminator/policy_accuracy`、`demo_accuracy`、logit、gradient penalty | LLC训练中的判别器信息；HLC冻结判别器，不重新训练它 |
| `HLC_Speed/<区间>/samples`、`mae_m_s`、`lateral_abs_m_s` | 按零速、(0,0.5)、[0.5,1)、[1,1.5)、[1.5,+∞)分组，统计全部有效LLC物理控制步；无样本区间不输出均值 |
| `HLC_Events/zero_quiet_fraction` | 零速有效物理步中的即时安静比例，仍不是持续停稳成功率 |
| `HLC_Events/stop_*`、`restart_*` | 指令切换事件的请求、未满足前置条件、完成、成功、失败、截断及待完成数量；成功率仅以已完成的有效事件为分母 |

事件统计跨rollout保留状态：走→0要求切换前实际水平速度≥0.1m/s，3秒内连续安静0.5秒且结束时已覆盖5秒路径窗口、末5秒路径≤0.1m。未走起来的请求记unqualified；换指令过早、timeout或重置记truncated，跌倒记failure。停后再起步要求前一有效停止事件成功且切换时安静，3秒内进入目标±0.1m/s、实际vx≥目标的50%并连续维持2秒；该速度事件统计还需结合视频确认迈步与位移。截断不计成功率，rollout边界不截断事件；`pending`是当前正在观测的事件数。短冒烟无完整事件时不输出成功率，不能解读为0%或100%。

`--json-logs`另外输出`F1_TRAIN`结构化日志；启动日志与checkpoint记录实际seed、参考依赖和控制配置。HLC训练帧数按高层决策计数，回合长度按底层步数/秒报告，比较时注意单位。跨LLC比较应以运动指标与视频为主，判别器奖励不能直接作为统一性能标尺。

## 上游来源

本项目基于[ASE（Adversarial Skill Embeddings）](https://github.com/nv-tlabs/ASE)适配F1，保留ASE/AMP与HRL实现。论文来源见[ASE项目页](https://xbpeng.github.io/projects/ASE/index.html)，仓库许可证见[LICENSE.txt](LICENSE.txt)。

仓库保留的原始Reallusion示例动作沿用上游“仅限非商业用途”的说明；F1使用的tracking NPZ与数据清单单独管理，参考清单及来源记录见上述F1数据说明和父工作区准备记录。
