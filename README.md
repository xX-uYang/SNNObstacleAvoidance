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
> 目前代码两条都实现了，**V1 主线以哪个为准尚未与队友定稿**，
> 见 `code/ann_baseline/docs/条件路由假数据联调说明.md`。

## 2. 仓库结构与模块分工

```
.
├── README.md                      本文件（项目门户）
├── docs/                          跨模块文档（分支规范等）
│
├── core/           ┐
├── scripts/        ├─ SNN 直觉通路（SNN 训练与推理实验）
├── document/       ┘
├── fix_imports.py
│
├── code/ann_baseline/   ANN 分析通路 + 融合 + 条件路由（完整工程，见其 README_CN.md）
├── runs/                ANN 实验记录（指标/日志；权重不入库，见 runs/README.md）
│
└── PythonAPI/examples/  CARLA 数据采集与环境搭建脚本（见其 README.md）
```

| 模块 | 目录 | 主要负责 |
| --- | --- | --- |
| SNN 直觉通路 | `core/`、`scripts/` | `@xX-uYang` |
| ANN 分析通路 / 融合 / 条件路由 | `code/ann_baseline/` | `@jk-Huo` |
| CARLA 数据采集与环境 | `PythonAPI/examples/` | `yzy` |
| 硬件与部署 | —（尚未入库） | 硬件组 |

另有**机器人避障数据集**的调研笔记（原 `Robot_Collision_Avoidance_Data` 分支）：
「考虑 CARLA 中虚拟仿真得到的驾驶数据的同时，我们尝试查看机器人避障的数据集」。
该分支只有一行 README，内容已并入此处，分支已废弃。

## 3. 分支说明（**先读这一节**）

2026 年整理时发现仓库里存在**两段互不相连的提交历史**：
`master`（SNN，2026-07-24 起）与 `ANN`/`Carla`（2026-03-25 起），
两者没有共同祖先，所以 `git log master` 里完全看不到 ANN 的代码，反之亦然。
整理时已用 `--allow-unrelated-histories` 把它们合并进 `main`。

| 分支 | 状态 | 说明 |
| --- | --- | --- |
| **`main`** | ✅ **推荐基线与集成分支** | 已整合 SNN + ANN + CARLA 三条工作线，91 个文件 |
| `ANN` | 活跃开发分支 | ANN 侧继续开发用；成果通过 PR 合回 `main` |
| `master` | 已并入 `main` | 内容与历史都已在 `main` 中，可只读保留 |
| `Carla` | 已并入 `main` | 同上 |
| `satan` | 与 `main` 旧版完全相同 | 无内容的占位分支，建议删除 |
| `Robot_Collision_Avoidance_Data` | 仅一行 README | 内容已并入本节，建议删除 |

> `main` 的默认分支地位**需要仓库所有者在 GitHub 设置里手动切换**
> （当前默认分支仍是 `master`）。切换前，新人点进仓库看到的仍是 SNN 那一半代码。

分支/提交/PR 的具体约定见 **[`docs/分支与协作规范.md`](docs/分支与协作规范.md)**。

## 4. 快速开始

### 4.1 环境

- Python 3.9.12
- PyTorch 2.4.1+cu124 / torchvision 0.19.1+cu124
- 参考 GPU：NVIDIA RTX 4060 Laptop 8GB（CPU 亦可跑通全部单元测试）

### 4.2 ANN 分析通路

```powershell
cd code\ann_baseline
pip install -r requirements.txt          # CPU / 基础依赖
pip install -r requirements-cuda.txt     # CUDA 版 PyTorch（按需）
pip install -r requirements-export.txt   # ONNX 导出（按需）
```

数据不入库（几百 MB）。把数据集放到**仓库根目录**的 `carla_data/` 下：

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

```powershell
# 训练（图像-only 基线）
python train.py --manifest data\carla_manifest.csv --output-dir runs\ann_image_v1 --pretrained --epochs 20

# 评估（测试集）
python evaluate.py --checkpoint runs\ann_image_v1\best.pt --manifest data\carla_manifest.csv --split test --output-dir runs\ann_image_v1_test

# 单图推理（可附带 SNN 三类概率）
python predict.py --checkpoint runs\ann_image_v1\best.pt --image ..\..\..\carla_data\Collision_Avoidance_System_Data\left\某张.png
python predict.py --checkpoint runs\ann_fusion_v1\best.pt --image demo.png --snn-left 0.15 --snn-straight 0.20 --snn-right 0.65

# 推理延迟 / ONNX 导出
python benchmark.py --checkpoint runs\ann_image_v1\best.pt --image 某张.png --device cuda
python export_onnx.py --checkpoint runs\ann_image_v1\best.pt --output exports\ann_image_v1.onnx

# 条件路由（SNN 事件触发）联调
python make_fake_routing_data.py
python validate_routing_data.py --help
python run_conditional_pipeline.py --help

# 冒烟测试
python -m unittest discover -s tests -v
```

更细的说明见 [`code/ann_baseline/README_CN.md`](code/ann_baseline/README_CN.md)。

### 4.3 SNN 直觉通路

见 `master` 分支的 `core/` 与 `scripts/`（约 22 个脚本），
**该部分目前还没有 README**，用法请直接问 `@xX-uYang`。
（把 SNN 侧 README 补上，是整理后遗留的待办之一。）

### 4.4 CARLA 数据采集

见 [`PythonAPI/examples/README.md`](PythonAPI/examples/README.md)。

## 5. 已有实验结论（第一轮，ANN-only）

来源：`runs/ANN第一轮训练结果.md`、`runs/ann_image_v1_test/metrics.json`

| 指标 | 数值 |
| --- | --- |
| 测试集 accuracy | **85.23%** |
| 测试集 balanced accuracy | **84.96%** |
| 左 / 直行 / 右 召回率 | 87.10% / 77.78% / 90.00% |
| 模型前向 P50 / P95 | 3.64 ms / 4.89 ms（RTX 4060, batch=1, 预热后） |

**必须说明**：`runs/fusion_plumbing_only*/` 下的融合结果（accuracy 54.55%）是
**用标签反推的假 SNN 概率**跑出来的，只能证明代码链路通，**不得作为融合性能引用**。
真实结论要等 SNN 侧产出真实输出后重跑。详见 [`runs/README.md`](runs/README.md)。

## 6. 仓库规范

### 6.1 不要入库的东西

`.gitignore` 已覆盖：`__pycache__/`、`.venv/`、`*.pt|pth|onnx|engine`、
`carla_data/`、图片视频、`*.zip`、`_tmp/`、`*.log`、各类凭据文件。
**`runs/` 下的 `*.json` / `*.csv` / `*.md` 等小体积结果仍会入库**（有记录价值），
只有权重等二进制产物被忽略。

### 6.2 换行符

仓库内文本一律以 LF 存储（`.gitattributes` 的 `* text=auto eol=lf`），
Windows 脚本保持 CRLF。**克隆时请勿修改 `core.autocrlf`**，
否则可能出现"全文件都被修改"的假 diff。

### 6.3 安全红线 🔒

- **绝不提交** 任何访问令牌、私钥、密码、凭据：GitHub / OpenAI / AWS / Slack token、
  `*.pem`、`*.key`、`id_rsa*`、`.env`、`service-account*.json` 等。
- 需要令牌时**只在本地环境变量里读**，例如 `GITHUB_TOKEN` → `process.env.GITHUB_TOKEN`，
  不要写进代码、不要写进命令行参数。
- 如不慎提交，**先吊销该令牌**再清理历史——只删文件不删历史等于没删。
- ⚠️ 本仓库目前是 **Public**，不要提交内部资料、个人绝对路径、客户/公司名称相关文件。

## 7. 当前进度与待办

**已完成**

- ANN 图像三分类基线训练/评估/推理/ONNX 导出全链路跑通，测试集 85.23%。
- SNN 先验融合的代码链路（`ConditionalANN`）跑通（假数据）。
- 条件路由层 `conditional_routing.py` + 数据校验 + 单测（12 项通过）。
- 仓库整理：合并两段历史、统一目录、补 `.gitignore`/`.gitattributes`、修正失效路径。

**进行中 / 未完成**

- ⚠️ SNN 与 ANN 的协作方式（事件触发路由 vs 特征融合）尚未定稿。
- ⚠️ `RouterConfig` 里的阈值（`min_calibrated_confidence=0.90`、
  `min_probability_margin=0.50`、`min_winner_membrane_ratio=1.00` 等）是**占位值**，
  必须用真实 SNN 输出重新标定。
- ⚠️ 尚无真实 SNN 输出；清单里 `snn_*` 三列仍为空。
- ⚠️ 数据只有单一批次（2026-04，625 张），另一批更大且类别不均衡的数据尚未纳入。
- ⚠️ SNN 侧没有 README；`core/config*.py` 内有硬编码本机路径。
- ⚠️ ROS 2 集成、Jetson/TensorRT 部署、闭环测试均未开始。
- ⚠️ 仓库仍无 LICENSE、无 CI、默认分支仍是 `master`。

## 8. 文档索引

| 文档 | 内容 |
| --- | --- |
| `docs/分支与协作规范.md` | 分支/提交/PR 约定，合并与冲突处理 |
| `code/ann_baseline/README_CN.md` | ANN 模块总览 |
| `code/ann_baseline/docs/SNN-ANN接口约定.md` | ROS 消息 / 融合输入输出约定 |
| `code/ann_baseline/docs/脚本说明与SNN-ANN融合指南.md` | 各脚本用途与融合用法 |
| `code/ann_baseline/docs/条件路由假数据联调说明.md` | 条件路由联调步骤 |
| `code/ann_baseline/data/README.md` | 数据清单列定义与路径约定 |
| `runs/README.md` | 哪些实验结论可以引用、哪些不可以 |
| `PythonAPI/examples/README.md` | CARLA 采集脚本用法 |
