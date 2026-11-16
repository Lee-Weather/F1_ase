# LLC停步数据候选集（2026-11-16）

本清单只调整参考数据与采样权重，未修改ASE网络、奖励、latent采样、控制或训练超参。
原dataset_f1.yaml及其24个文件逐项SHA256未变。新候选入口：dataset_f1_stop_data.yaml。

训练共36条，按(n-1)/50计203.26秒。全片保留连续运动；没有裁剪拼接、冻结末帧或人工延长站立尾段。

权重分配：原有其他运动60%、新增其他运动10%、减速→站立25%、A1站立5%。这是片段采样概率，不是文件数/帧数比例；当前MotionLib权重同时影响示范采样及参考起步采样。

## 完整训练名单

| # | 文件 | 分组 | 采样概率 | 时长(s) |
|---:|---|---|---:|---:|
| 1 | A1_stand_weichai_f1_29.npz | 站立 | 5.00000% | 2.98 |
| 2 | A2_sway_weichai_f1_29.npz | 原有其他运动 | 0.86331% | 12.00 |
| 3 | B1_stand_to_walk_weichai_f1_29.npz | 原有其他运动 | 2.15827% | 6.20 |
| 4 | B2_walk_to_stand_weichai_f1_29.npz | 减速→站立 | 6.57895% | 5.66 |
| 5 | B2t2_walk_to_stand_weichai_f1_29.npz | 减速→站立 | 5.26316% | 4.88 |
| 6 | B3_walk1_weichai_f1_29.npz | 原有其他运动 | 4.31655% | 7.60 |
| 7 | B4_stand_to_walk_back_weichai_f1_29.npz | 原有其他运动 | 2.15827% | 7.54 |
| 8 | B5_walk_backwards_weichai_f1_29.npz | 原有其他运动 | 2.15827% | 11.52 |
| 9 | C2_Run_to_stand_f1.npz | 减速→站立 | 6.57895% | 5.30 |
| 10 | C3_Run_f1.npz | 原有其他运动 | 4.31655% | 1.68 |
| 11 | C4_Run_to_walk_f1.npz | 原有其他运动 | 2.15827% | 3.52 |
| 12 | C5_Walk_to_run_f1.npz | 原有其他运动 | 3.02158% | 5.40 |
| 13 | C6_stand_to_run_backwards_weichai_f1_29.npz | 原有其他运动 | 1.72662% | 7.46 |
| 14 | C7_run_backwards_weichai_f1_29.npz | 原有其他运动 | 1.72662% | 3.08 |
| 15 | M2_C1_stand_to_run_weichai_f1_29.npz | 原有其他运动 | 2.58993% | 4.08 |
| 16 | M2_C3_run_weichai_f1_29.npz | 原有其他运动 | 4.31655% | 2.70 |
| 17 | M2_C4_run_to_walk_weichai_f1_29.npz | 原有其他运动 | 2.15827% | 3.84 |
| 18 | M2_C5_walk_to_run_weichai_f1_29.npz | 原有其他运动 | 3.02158% | 3.54 |
| 19 | S007_QkWalk1_weichai_f1_29.npz | 原有其他运动 | 2.58993% | 2.54 |
| 20 | S008_Run1_weichai_f1_29.npz | 原有其他运动 | 4.31655% | 1.78 |
| 21 | S009_Sprint1_weichai_f1_29.npz | 原有其他运动 | 3.45324% | 2.18 |
| 22 | LAFAN_run_moderate_s2_f1.npz | 原有其他运动 | 4.31655% | 2.96 |
| 23 | LAFAN_run_fast_s4_f1.npz | 原有其他运动 | 4.31655% | 2.96 |
| 24 | LAFAN_walk_s1_f1.npz | 原有其他运动 | 4.31655% | 7.96 |
| 25 | accad_walking_extra_50hz/Female1_B10_walk_turn_left_45_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 7.90 |
| 26 | accad_walking_extra_50hz/Female1_B11_walk_turn_left_135_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 9.34 |
| 27 | accad_walking_extra_50hz/Female1_B14_walk_turn_right_135_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 8.94 |
| 28 | accad_walking_extra_50hz/Female1_B15_walk_turn_around_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 10.58 |
| 29 | accad_walking_extra_50hz/Female1_B6_walk_backwards_to_stand_weichai_f1_29.npz | 减速→站立 | 6.57895% | 8.16 |
| 30 | accad_walking_extra_50hz/Female1_B9_walk_turn_left_90_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 7.08 |
| 31 | accad_walking_extra_50hz/Male2_B10_turn_left_45_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 5.96 |
| 32 | accad_walking_extra_50hz/Male2_B11_turn_left_135_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 6.48 |
| 33 | accad_walking_extra_50hz/Male2_B14_turn_right_45_t2_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 5.78 |
| 34 | accad_walking_extra_50hz/Male2_B14_turn_right_45_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 4.08 |
| 35 | accad_walking_extra_50hz/Male2_B4_stand_to_walk_backwards_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 4.58 |
| 36 | accad_walking_extra_50hz/Male2_B9_turn_left_90_weichai_f1_29.npz | 新增其他运动 | 0.90909% | 5.02 |

## 新增数据筛选

26条原始120Hz qpos先用既有mujoco_to_npz.py及当前F1 MJCF转换为50Hz tracking NPZ。按当前碰撞网格计算每帧左右脚中最低碰撞点高度。新增纯行走仅选择至少50%帧有脚部最低点≤2cm的轨迹。这是保守选材阈值，不是接触力、脚滑或动力学验收；原24条作为对照保留，未按此纯行走阈值重筛。

| 新增文件 | 处理 | 脚部近地帧比例 | 原因 |
|---|---|---:|---|
| Female1_B10_walk_turn_left_45_weichai_f1_29.npz | 入选 | 89.9% | 纳入新清单 |
| Female1_B11_walk_turn_left_135_weichai_f1_29.npz | 入选 | 89.5% | 纳入新清单 |
| Female1_B12_walk_turn_right_90_weichai_f1_29.npz | 暂缓 | 3.1% | 近地比例不足，待修复/复核 |
| Female1_B13_walk_turn_right_45_weichai_f1_29.npz | 暂缓 | 34.2% | 近地比例不足，待修复/复核 |
| Female1_B14_walk_turn_right_135_weichai_f1_29.npz | 入选 | 89.3% | 纳入新清单 |
| Female1_B15_walk_turn_around_weichai_f1_29.npz | 入选 | 99.8% | 纳入新清单 |
| Female1_B16_walk_turn_change_direction_weichai_f1_29.npz | 暂缓 | 29.2% | 近地比例不足，待修复/复核 |
| Female1_B2_walk_to_stand_t2_weichai_f1_29.npz | 去重 | — | 与B2t2_walk_to_stand_weichai_f1_29.npz重复 |
| Female1_B5_walk_backwards_weichai_f1_29.npz | 去重 | — | 与B5_walk_backwards_weichai_f1_29.npz重复 |
| Female1_B6_walk_backwards_to_stand_weichai_f1_29.npz | 入选 | 58.2% | 纳入新清单 |
| Female1_B7_walk_backwards_turn_forwards_weichai_f1_29.npz | 暂缓 | 42.3% | 近地比例不足，待修复/复核 |
| Female1_B9_walk_turn_left_90_weichai_f1_29.npz | 入选 | 96.3% | 纳入新清单 |
| Male1_Walk_B10_turn_left_45_weichai_f1_29.npz | 暂缓 | 28.0% | 近地比例不足，待修复/复核 |
| Male1_Walk_B13_turn_right_45_weichai_f1_29.npz | 暂缓 | 6.7% | 近地比例不足，待修复/复核 |
| Male1_Walk_B15_turn_around_weichai_f1_29.npz | 暂缓 | 38.4% | 近地比例不足，待修复/复核 |
| Male1_Walk_B16_turn_change_weichai_f1_29.npz | 暂缓 | 25.2% | 近地比例不足，待修复/复核 |
| Male1_Walk_B4_stand_to_walk_back_weichai_f1_29.npz | 暂缓 | 19.0% | 近地比例不足，待修复/复核 |
| Male2_B10_turn_left_45_weichai_f1_29.npz | 入选 | 97.3% | 纳入新清单 |
| Male2_B11_turn_left_135_weichai_f1_29.npz | 入选 | 100.0% | 纳入新清单 |
| Male2_B13_turn_right_90_weichai_f1_29.npz | 暂缓 | 24.9% | 近地比例不足，待修复/复核 |
| Male2_B14_turn_right_45_t2_weichai_f1_29.npz | 入选 | 64.5% | 纳入新清单 |
| Male2_B14_turn_right_45_weichai_f1_29.npz | 入选 | 100.0% | 纳入新清单 |
| Male2_B15_turn_around_weichai_f1_29.npz | 暂缓 | 38.7% | 近地比例不足，待修复/复核 |
| Male2_B4_stand_to_walk_backwards_weichai_f1_29.npz | 入选 | 91.7% | 纳入新清单 |
| Male2_B5_walk_backwards_weichai_f1_29.npz | 暂缓 | 0.0% | 近地比例不足，待修复/复核 |
| Male2_B9_turn_left_90_weichai_f1_29.npz | 入选 | 100.0% | 纳入新清单 |

暂缓的12个50Hz文件保留在项目czy/data/reference_review/accad_walking_extra_50hz_deferred/；2个重复的转换副本删除，外部原始文件和原训练文件保留。原始来源：E:\UMR\umr_data\weichai_f1_29_accad_walking_extra

逐文件来源/哈希/近地/末速统计：czy/data/reference_review/llc_stop_data.json；完整训练名单CSV：czy/data/reference_review/llc_stop_data.csv。

## 复现（从E:/F1/F1_ase_h运行）

```powershell
python czy/f1_wbt/scripts/mujoco_to_npz.py --input_dir "E:\UMR\umr_data\weichai_f1_29_accad_walking_extra" --output_dir F1_ase/ase/data/motions/f1/accad_walking_extra_50hz --model F1_ase/ase/data/assets/f1/mjcf/f1_29dof_scene.xml --fps 50
python czy/f1_wbt/scripts/prepare_f1_stop_data.py --source "E:\UMR\umr_data\weichai_f1_29_accad_walking_extra"
```

本次仅数据准备，训练轮数/资源未确定，未启动新训练。

201.5真实F1环境CPU加载检查通过：36条文件哈希全部一致、关节状态[36,29]及中点插值有限，归一化采样概率60%/10%/25%/5%一致。仿真步数0、优化器更新0。
