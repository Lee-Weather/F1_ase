# LAFAN能力补齐参考子集

2026-11-18筛选：4条，35.94秒；50Hz，29关节，53个刚体。这里只保存ASE可读的压缩NPZ与来源说明；未改变7.0/7.1清单或默认训练入口，尚未用于LLC训练。

| 文件 | 内容 | 原BVH / 原帧区间（左闭右开） | F1时长(s) | F1均速(m/s) | 根部及脚部低速最长(s) |
|---|---|---|---:|---:|---:|
| LAFAN_slow_walk2_s4_f1.npz | 低速步行/转向 | walk2_subject4.bvh [1350,1650) | 9.96 | 0.215 | 0.00 |
| LAFAN_slow_walk1_s2_f1.npz | 低速换向/侧后步 | walk1_subject2.bvh [90,330) | 7.96 | 0.165 | 0.32 |
| LAFAN_slow_walk3_s3_f1.npz | 低速步行/转向 | walk3_subject3.bvh [2490,2790) | 9.96 | 0.204 | 0.82 |
| LAFAN_pause_restart_walk3_s3_f1.npz | 运动→停顿→再运动 | walk3_subject3.bvh [481,724) | 8.06 | 0.170 | 3.10 |

源文件按30Hz帧号裁剪，经BVH→SMPL-X Neutral→本地UMR既有F1配置→50Hz tracking转换；没有时间缩放、冻结站立尾帧、拼接或人为拉直路径。NPZ重压缩前后每个数组逐值相同。均速是参考运动学数值，不是策略表现或恒速指令；慢走片段可能包含转向、侧向/后向成分。

已检查有限值、四元数、关节顺序/限位、基座高度/倾斜、脚部几何近地及穿地，并审阅骨架抽帧。本批没有入选恢复步；两段恢复候选近地支持不足，已暂缓。停顿使用0.3秒平滑根部速度<0.1m/s且双脚最大刚体速度<0.15m/s统计，不能等同动力学站稳。

原始包、裁剪BVH、SMPL-X、UMR qpos、未入选文件与审查材料保存在仓库外：`E:/F1/F1_ase_h/czy/npz/lafan1.zip`、`czy/npz/lafan1/`、`czy/data/reference_review/lafan/curated/`（均相对项目父目录）；仓库内不重复保存。复现脚本为父目录下`czy/f1_wbt/scripts/curate_lafan_capabilities.py`及`package_lafan_capabilities.py`；`selection.json`保留初选帧段，`review.json`记录全部通过/暂缓原因，`approved.json`记录最终入选ID。

来源：[Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)，许可CC BY-NC-ND 4.0；BVH→SMPL-X映射复用jaraujo98/lafan_to_smplx，MIT归属保留在转换脚本。逐文件原始来源与最终SHA256见[manifest.json](manifest.json)。后续组装正式训练集时再确定采样权重；本目录不是已确认的exp7.3训练集。
