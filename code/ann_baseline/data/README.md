# data/ —— 数据清单（manifest）

本目录**只放清单（CSV），不放图片**。图片是几百 MB 的原始数据，按 `.gitignore`
规则不入库（见仓库根 `.gitignore` 的 `carla_data/` 与 `data/**/*.png`）。

## 这两个文件是什么

| 文件 | 用途 | 能否用于报告性能 |
| --- | --- | --- |
| `carla_manifest.csv` | **冻结清单**。定义了本阶段 ANN 三分类的 train/val/test 划分，`snn_*` 三列全空（即"无 SNN 先验"的图像-only 基线）。 | ✅ 可以 |
| `carla_manifest_with_fake_snn.csv` | 联调清单。`snn_*` 三列是由 **真实标签反推出来的假概率**，只用来验证"融合/路由代码能否跑通"。 | ❌ **绝对不可以** |

> 假 SNN 清单的 `snn_source` 列固定为 `fake_debug_only_ground_truth_derived`，
> 用它跑出来的精度天然虚高，任何文档、汇报中都不得引用为融合性能。

`carla_manifest.csv` 是**冻结**的：所有人在它上面跑出的指标才可比。
不要用重新生成的清单覆盖它（见文末"重新生成"）。

## 列定义

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `image_path` | 相对路径 | 相对**本文件所在目录**（即 `code/ann_baseline/data/`）解析，见下方"路径约定" |
| `split` | `train` / `val` / `test` | 划分结果 |
| `label` | `0` / `1` / `2` | `0=left`、`1=straight`、`2=right`（与 `src/ann_baseline/constants.py` 一致） |
| `class_name` | `left` / `straight` / `right` | 标签的原始字符串 |
| `episode_id` | `time_group_00N` | **时间分组 ID**，来自按时间间隔切出的连续片段 |
| `capture_timestamp` | ISO 8601 | 图像文件名中解析出的采集时间（`%Y%m%d_%H%M%S_%f`） |
| `snn_left` / `snn_straight` / `snn_right` | float | SNN 三类概率；冻结清单中为空 |
| `snn_source` | str | SNN 输出来源标记；真实接入时应为 SNN 侧写出的版本号 |
| `steering` | float | 可选转向回归标签；当前为空（仅图像分类任务） |

**划分是按 `episode_id`（时间块）整块划分的，不是按帧随机划分**——同一段连续
视频的相邻帧高度相似，若随机划分会造成训练/测试泄漏，指标虚高。

## 路径约定（重要）

`image_path` 是相对路径，基准是**本目录**，写法形如：

```
../../../carla_data/Collision_Avoidance_System_Data/straight/STRAIGHT_2297_20260404_030659_042573.png
```

即约定数据集放在**仓库根目录下的 `carla_data/`**：

```
SNNObstacleAvoidance/            <- 仓库根
├── carla_data/                  <- 数据集（Git 忽略，不入库）
│   └── Collision_Avoidance_System_Data/
│       ├── left/     (204 张)
│       ├── straight/ (204 张)
│       └── right/    (217 张)
└── code/ann_baseline/
    └── data/carla_manifest.csv  <- 清单
```

数据集不在仓库里时，用 Windows 目录联接（junction，无需管理员权限、不占空间）
把已有数据挂进来即可：

```cmd
mklink /J E:\SNNObstacleAvoidance\carla_data E:\Intuition\carla_data
```

删除联接用 `rmdir E:\SNNObstacleAvoidance\carla_data`（只会删链接，不会删真实数据）。

## 当前清单统计（625 行）

| split | left | straight | right | 合计 |
| --- | --- | --- | --- | --- |
| train | 147 | 144 | 156 | 447 |
| val | 26 | 33 | 31 | 90 |
| test | 31 | 27 | 30 | 88 |
| **合计** | **204** | **204** | **217** | **625** |

对应 `runs/ann_image_v1_test/metrics.json` 中 ANN-only 基线
（accuracy 85.23% / balanced accuracy 84.96%）。

## 校验

```powershell
# 1) 清单能否加载、图片是否都在（路径写错会直接抛 FileNotFoundError）
python -c "from pathlib import Path; from ann_baseline.data import load_manifest; print(len(load_manifest(Path('data/carla_manifest.csv'))))"

# 2) 条件路由用的三路证据数据是否合规
python validate_routing_data.py --help
```

## 重新生成（谨慎）

```powershell
python prepare_carla_manifest.py `
  --dataset-root ..\..\..\carla_data\Collision_Avoidance_System_Data `
  --output data\carla_manifest_new.csv `
  --train-ratio 0.70 --val-ratio 0.15 --new-episode-gap-seconds 20.0 --seed 2026
```

**注意**：脚本内部用随机搜索做类别均衡，即使 `--seed` 相同，输入图片集合或
分组结果一变，划分就会变。所以：

- 想复现已有的 85.23% 基线 → **必须用仓库里这份冻结的 `carla_manifest.csv`**；
- 想换数据批次 → 生成新文件（如 `carla_manifest_v2.csv`），**另起一个 output-dir**
  重跑，不要覆盖旧清单、不要把新结果和旧结果放在一起比。

## 已知限制 / 待办

1. 当前清单只覆盖一批数据（`Collision_Avoidance_System_Data`，采集于 2026-04，
   625 张）。本机另外还有一批更晚、更大且**类别不均衡**的数据
   （`carla_data8.6/episode_20260805_*`，共 1834 张：left 355 / straight 1156 /
   right 323），尚未纳入，需要先和 CARLA 负责人确认它是不是本阶段要用的冻结批次。
2. `snn_*` 三列目前为空，等 SNN 侧产出真实输出后，需要通过
   `merge_snn_outputs.py` 回填，才能跑真正的融合/路由实验。
3. `steering` 列为空，转向回归分支只是预留。
