# CMU能力补齐参考子集

2026-11-18筛选入库4条，共37.64秒；F1 tracking为50Hz、29关节、53刚体。只保存压缩NPZ、README和来源清单；未用于新训练。

| 文件 | 内容 | 源帧区间（120Hz，左闭右开） | F1时长(s) | F1均速(m/s) | 根部及双脚同时低速最长(s) |
|---|---|---|---:|---:|---:|
| CMU_91_10_f1.npz | 慢走风格片段 | [1012,1972) | 7.98 | 0.241 | 1.12 |
| CMU_91_27_f1.npz | 另一段慢走风格 | [1048,2008) | 7.98 | 0.338 | 0.16 |
| CMU_82_08_f1.npz | 站立后自然起步并行走 | [0,1699) | 14.14 | 0.227 | 8.48 |
| CMU_82_14_f1.npz | 行走后减速 | [0,907) | 7.54 | 0.396 | 2.22 |

源为AMASS CMU的SMPL-X Neutral（16维betas，现有UMR采用前10维）；保留原120Hz源时间，经现有UMR F1配置重定向，再转换50Hz。慢走仅选代表性连续片段，起步/减速保留完整源过程。没有冻结尾帧、拼接、时间缩放或拉直路径。原始路径、完整源SHA256、裁剪帧与最终NPZ哈希见[manifest.json](manifest.json)。

均速是F1参考运动学数据，不是指令速度或策略表现。根部/双脚低速使用0.3秒平滑根速<0.1m/s且最大脚部刚体速度<0.15m/s，不能等同动力学站稳。没有因文件名含stop就认定充分停稳；本批不计作受推恢复示范。

入库前核对有限值、单位四元数、关节名称顺序与限位、基座高度/倾斜终止冲突、脚部碰撞几何近地/明显穿地，并检查源人体及F1骨架抽帧。通过筛选不等于动力学无滑动或LLC已学会。NPZ压缩前后数组逐值相同。

原始包`E:/F1/F1_ase_h/czy/npz/CMU.tar.bz2`及全部选材、中间结果、暂缓数据位于训练仓库外：`E:/F1/F1_ase_h/czy/data/reference_review/cmu/curated/`；不完整解压CMU，不复制源NPZ、SMPL-X裁剪、qpos或审查图进Git。该目录的source_review.json/review.json记录初选与暂缓，approved.json记录入选；reader_check.json记录现有ASE读取器验证。

复现工具在项目父目录`czy/f1_wbt/scripts/`：extract_cmu_priorities.py、curate_cmu_capabilities.py；UMR提交与配置/脚本哈希记录于manifest。原7.0/7.1清单及默认训练入口未更改，新训练集组合和采样权重另定。

人体来源：[AMASS](https://amass.is.tue.mpg.de/)，动作描述：[CMU官方索引](https://mocap.cs.cmu.edu/search.php)。
