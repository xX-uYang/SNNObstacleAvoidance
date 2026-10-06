# SNNObstacleAvoidance

> 基于**脉冲神经网络（SNN）直觉推理**的自动驾驶紧急避障系统。
> 用 SNN 提供"低延迟直觉"，用 ANN 提供"高精度分析"，再经安全仲裁与 CARLA 闭环仿真构成完整链路。

---

## 1. 系统概览

```
                       ┌──────────────────────────────┐
   CARLA 图像 ─────────┤  快通路：SNN 直觉推理         │──┐
   (前视相机)           │  低延迟、低功耗，可独立下结论  │  │
                       └──────────────────────────────┘  │
                                                          ├──► 安全仲裁 ──► 车辆控制
                       ┌──────────────────────────────┐  │      │
        同一帧图像 ────┤  慢通路：ANN 分析             │──┘      │
                       │  MobileNetV3-Small 三分类     │         │
                       │  left / straight / right      │◄────────┘
                       └──────────────────────────────┘   必要时才唤起
```

**关键设计意图**：两条通路是**并行**的，不是串行流水线。
ANN 的价值在于「高精度复核 + 在 SNN 不确定时补位」；算力节省来自
**SNN 能独立下结论时就不调用 ANN**，而不是靠"ANN 跑得快"。

> 说明：关于 SNN 与 ANN 的具体协作方式，团队文档中存在两种候选设计——
> ①特征/决策级融合（`ConditionalANN(use_snn_prior=True)`）；
> ②事件触发式条件路由（`SNN_DIRECT / CALL_ANN / SAFE_FALLBACK`）。
> 目前代码两条都实现了，**V1 主线以哪个为准尚未与队友定稿**，见 `code/ann_baseline/docs/`。

## 2. 仓库现状：分支对照

本仓库由多人分工开发，历史上有过两段互不相连的提交历史（2026-03 与 2026-07 各初始化过一次），
目前**尚未合并**。各分支状态如下：

| 分支 | 负责模块 | 内容 | 状态 |
| --- | --- | --- | --- |
| `master` | SNN 训练 | `core/`（SNN 模型与数据集）、`scripts/`（约 22 个训练/测试脚本） | 有代码，**无 README**，含硬编码本机路径 |
| `ANN` | **ANN 分析通路 / 融合与路由** | `code/ann_baseline/`（完整 ANN 工程）、`runs/`（实验记录） | **当前最活跃**，接口文档见 `code/ann_baseline/docs/` |
| `Carla` | 数据采集 | `PythonAPI/examples/` 下的采集与场景生成脚本 | 可用，数据规范见该分支 `README.md` |
| `main` | 集成 | 仅一个 2 行 README | 占位，未启用 |
| `satan` | — | 与 `main` 完全相同 | 占位，建议删除 |
| `Robot_Collision_Avoidance_Data` | 数据调研 | 仅一个 README（机器人避障数据集调研笔记） | 占位，建议并入 `main` 后删除 |

**里程碑**：本仓库至今只有 1 个 PR（#1 `Update README.md`，2026-03-30，仅改 README），
ANN 的代码是 2026-08-03 直接推到 `ANN` 分支的，**尚未通过 PR 合入任何集成分支**。

## 3. 快速开始（ANN 部分）

### 3.1 环境

- Python 3.9.12
- PyTorch 2.4.1+cu124 / torchvision 0.19.1+cu124
- 参考 GPU：NVIDIA RTX 4060 Laptop 8GB（CPU 亦可跑通全部单元测试）

```powershell
cd code\ann_baseline
pip install -r requirements.txt          # CPU / 基础依赖
pip install -r requirements-cuda.txt     # CUDA 版 PyTorch（按需）
pip install -r requirements-export.txt   # ONNX 导出（按需）
```

### 3.2 数据

图片不入库（几百 MB）。把数据集放到**仓库根目录**的 `carla_data/` 下：

```
SNNObstacleAvoidance/carla_data/Collision_Avoidance_System_Data/{left,straight,right}/
```

数据已在别处时，用目录联接挂进来即可（免管理员、不占额外空间）：

```cmd
mklink /J E:\SNNObstacleAvoidance\carla_data E:\Intuition\carla_data
```

清单 `code/ann_baseline/data/carla_manifest.csv` 是**冻结**的划分（625 张，
train 447 / val 90 / test 88），它是所有指标可比的前提。
**列定义、路径约定、以及"为什么不能重新生成"见 `code/ann_baseline/data/README.md`。**

### 3.3 训练 / 评估 / 推理

```powershell
# 训练（图像-only 基线）
python train.py --manifest data\carla_manifest.csv --output-dir runs\ann_image_v1 --pretrained --epochs 20

# 评估（测试集）
python evaluate.py --checkpoint runs\ann_image_v1\best.pt --manifest data\carla_manifest.csv --split test --output-dir runs\ann_image_v1_test

# 单图推理（可附带 SNN 三类概率）
python predict.py --checkpoint runs\ann_image_v1\best.pt --image ..\..\..\carla_data\Collision_Avoidance_System_Data\left\某张.png
python predict.py --checkpoint runs\ann_fusion_v1\best.pt --image demo.png --snn-left 0.15 --snn-straight 0.20 --snn-right 0.65

# 推理延迟
python benchmark.py --checkpoint runs\ann_image_v1\best.pt --image 某张.png --device cuda

# ONNX 导出（部署用）
python export_onnx.py --checkpoint runs\ann_image_v1\best.pt --output exports\ann_image_v1.onnx
```

### 3.4 条件路由（SNN 事件触发）联调

```powershell
python make_fake_routing_data.py                     # 生成假三路证据数据
python validate_routing_data.py --help               # 数据体检
python run_conditional_pipeline.py --help            # 离线评估路由：只在 CALL_ANN 时才调用 ANN
python -m unittest discover -s tests -v              # 12 项单元测试
```

### 3.5 冒烟测试

```powershell
python -m unittest discover -s tests -v
```

## 4. 已有实验结论（第一轮，ANN-only）

来源：`runs/ANN第一轮训练结果.md`、`runs/ann_image_v1_test/metrics.json`

| 指标 | 数值 |
| --- | --- |
| 测试集 accuracy | **85.23%** |
| 测试集 balanced accuracy | **84.96%** |
| 左 / 直行 / 右 召回率 | 87.10% / 77.78% / 90.00% |
| 模型前向 P50 / P95 | 3.64 ms / 4.89 ms（RTX 4060, batch=1, 预热后） |

**必须说明**：`runs/fusion_plumbing_only*/` 下的融合结果（accuracy 54.55%）是
**用标签反推的假 SNN 概率**跑出来的，只能证明代码链路通，**不得作为融合性能引用**。
真实结论要等 SNN 侧产出真实输出后重跑。

## 5. 目录结构（`ANN` 分支）

```
.
├── README.md                    本文件
├── .gitignore / .gitattributes  忽略规则 / 换行符规则
├── code/ann_baseline/           ANN 工程（唯一权威目录）
│   ├── train.py / evaluate.py / predict.py / benchmark.py / export_onnx.py
│   ├── prepare_carla_manifest.py     从图片目录生成清单
│   ├── merge_snn_outputs.py          把 SNN 输出回填进清单
│   ├── run_conditional_pipeline.py   条件路由离线评估
│   ├── validate_routing_data.py      路由数据体检
│   ├── make_fake_routing_data.py     假路由数据（联调用）
│   ├── src/ann_baseline/             核心库
│   │   ├── model.py                  ConditionalANN (MobileNetV3-Small)
│   │   ├── data.py / constants.py / metrics.py / checkpoint.py
│   │   ├── conditional_routing.py    路由决策 SNN_DIRECT/CALL_ANN/SAFE_FALLBACK
│   │   └── conditional_data.py       三路证据对齐与校验
│   ├── docs/                         接口约定与联调说明
│   ├── data/                         清单 CSV + 数据说明
│   ├── ros_msgs/SnnPrior.msg         SNN→ANN 的 ROS 2 消息定义
│   └── tests/                        单元测试
└── runs/                        实验记录（只保留小体积文本结果，权重不上传）
```

## 6. 仓库规范

### 6.1 提交与分支

- 分支：`master`(SNN) / `ANN` / `Carla` 各自开发；**集成分支尚未定稿**（计划启用 `main`）。
- 提交信息：`类型(范围): 摘要`，例如 `feat(ann): ...`、`fix(data): ...`、`docs: ...`。
- 提交前自查：

  ```powershell
  git status                 # 确认没有误加 __pycache__ / *.pt / 数据集
  python -m unittest discover -s tests -v
  ```

### 6.2 不要入库的东西

`.gitignore` 已覆盖：`__pycache__/`、`*.pyc`、`.venv/`、`*.pt|pth|onnx|engine`、
`carla_data/`、`*.zip`、`_tmp/`、`*.log`。
**`runs/` 下的 `*.json` / `*.csv` / `*.md` 等小体积结果仍会入库**（有记录价值），
只有权重等二进制产物被忽略。

### 6.3 换行符

仓库内文本一律以 LF 存储（`.gitattributes` 的 `* text=auto eol=lf`），
Windows 脚本保持 CRLF。**克隆时请勿修改 `core.autocrlf`**，
否则可能出现"全文件都被修改"的假 diff。

### 6.4 安全红线 🔒

- **绝不提交** 任何访问令牌、私钥、密码、凭据：GitHub / OpenAI / AWS / Slack token、
  `*.pem`、`*.key`、`id_rsa*`、`.env`、`service-account*.json` 等。
- 需要令牌时**只在本地环境变量里读**，例如 `GITHUB_TOKEN` → `process.env.GITHUB_TOKEN`，不要写进代码。
- 如不慎提交，立即**吊销该令牌**并清理历史，清理前不要只删文件了事——历史里还在。

## 7. 当前进度与待办

**已完成**

- ANN 图像三分类基线训练/评估/推理/ONNX 导出全链路跑通，测试集 85.23%。
- SNN 先验融合的代码链路（`ConditionalANN`）跑通（假数据）。
- 条件路由层 `conditional_routing.py` + 数据校验 + 单测（12 项通过）。

**进行中 / 未完成**

- ⚠️ SNN 与 ANN 的协作方式（事件触发路由 vs 特征融合）尚未定稿。
- ⚠️ `RouterConfig` 里的阈值（`min_calibrated_confidence=0.90`、
  `min_probability_margin=0.50`、`min_winner_membrane_ratio=1.00` 等）是**占位值**，
  必须用真实 SNN 输出重新标定。
- ⚠️ 尚无真实 SNN 输出；`snn_*` 三列仍为空。
- ⚠️ 数据只有单一批次（2026-04，625 张），另一批更大且类别不均衡的数据尚未纳入。
- ⚠️ ROS 2 集成、Jetson/TensorRT 部署、闭环测试均未开始。
- ⚠️ 三个分支历史尚未合并；本仓库仍是 Public，注意不要提交内部资料与本机绝对路径。

## 8. 接口文档索引

| 文档 | 内容 |
| --- | --- |
| `code/ann_baseline/docs/SNN-ANN接口约定.md` | ROS 消息 / 融合输入输出约定 |
| `code/ann_baseline/docs/脚本说明与SNN-ANN融合指南.md` | 各脚本用途与融合用法 |
| `code/ann_baseline/docs/条件路由假数据联调说明.md` | 条件路由联调步骤 |
| `code/ann_baseline/data/README.md` | 数据清单列定义与路径约定 |
| `code/ann_baseline/ros_msgs/SnnPrior.msg` | `SnnPrior` 消息定义 |
