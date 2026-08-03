# 脚本说明与 SNN–ANN 融合实施指南

## 一、先把 ANN 和 SNN 说清楚

ANN（Artificial Neural Network，人工神经网络）是一个大类。当前代码中的 ANN 是以 MobileNetV3-Small 为图像骨干的卷积神经网络：每一层使用连续实数激活，读入一张 RGB 图像后一次前向计算，输出左、直行、右三个分数。对本项目而言，ANN 的优势是能利用图像中的道路纹理、车道线、障碍物和全局场景；不足是每一帧通常都要做密集计算，而且单帧模型不天然表达时间过程。

SNN（Spiking Neural Network，脉冲神经网络）把信息表示为随时间出现的离散脉冲，并维护膜电位等内部状态。它更自然地处理时间变化和事件流，在合适的神经形态硬件上可能具备低延迟、低功耗优势。由于发放函数不可导，常用替代梯度训练。SNN 并不天然比 ANN 更准确，也不等于只要换成脉冲就一定省电；优势需要结合稀疏事件、时间编码、部署硬件和实测延迟/功耗才能成立。

适合入门的原始资料：

- [PyTorch 迁移学习官方教程](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial)
- [Torchvision MobileNetV3-Small 官方说明](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html)
- [snnTorch 官方教程入口](https://snntorch.readthedocs.io/en/stable/readme.html)
- [snnTorch 替代梯度说明](https://snntorch.readthedocs.io/en/stable/snntorch.surrogate.html)
- [Neftci 等：Surrogate Gradient Learning in Spiking Neural Networks](https://arxiv.org/abs/1901.09948)

## 二、当前代码已经形成的流程

```text
carla_data
   │
   ├─ 数据审计 ──> 质量报告、重复检查、时间分布
   │
   └─ 按时间组划分 ──> carla_manifest.csv
                             │
                  ┌──────────┴──────────┐
                  │                     │
              图像 ANN              SNN 输出
                  │                     │
                  └──── 特征/决策融合 ──┘
                             │
                     左 / 直行 / 右
                             │
                 下一阶段：连续 steer 回归
```

当前固定类别顺序为：

```text
0 = left
1 = straight
2 = right
```

SNN 条件输入顺序为：

```text
[p_left, p_straight, p_right, available_mask]
```

`available_mask=0` 表示该帧没有同步、可用的 SNN 输出，此时三个数保持中性值 `1/3`，ANN 学习退回图像单模态判断。

## 三、每个脚本做什么

### 数据准备与检查

`audit_carla_data.py`

- 检查图片能否打开、分辨率与色彩模式是否一致。
- 计算 SHA-256，找完全重复文件。
- 计算 dHash，粗筛非常相似的画面。
- 解析文件名时间戳，统计采集跨度和相邻时间间隔。
- 输出逐图 CSV 与汇总 JSON。

`prepare_carla_manifest.py`

- 专门读取当前的平铺结构：`carla_data/left`、`straight`、`right`。
- 按文件名时间戳排序。
- 默认相邻记录间隔超过 20 秒才新建一个时间组。
- 整个时间组只进入 train、val、test 中的一个，避免连续帧泄漏。
- 生成训练程序统一读取的 `carla_manifest.csv`。

`build_manifest.py`

- 服务于更规范的未来数据结构：`dataset/episode_001/{left,straight,right}`。
- 按完整 episode 划分，适合后续重新采集 CARLA 数据时使用。
- 当前这批平铺数据应使用 `prepare_carla_manifest.py`，不要使用它。

`generate_fake_snn_outputs.py`

- 根据真实标签生成带噪声的三分类“假 SNN 概率”，并随机制造一部分缺失输出。
- 目的仅是测试 CSV 字段、归一化、缺失回退和融合分支。
- 因为它读取了真实标签，所以存在刻意的数据泄漏；由它产生的融合准确率没有实验意义。

`merge_snn_outputs.py`

- 等黄祺茵导出真实 SNN 结果后，按 `image_path` 把三类概率合并到 ANN 清单。
- 默认要求每张图都有 SNN 结果；只有做缺失模态消融时才使用 `--allow-missing`。

`make_toy_dataset.py`

- 生成极小的人工道路图，用来检查程序能否启动。
- 它不是 CARLA 数据，也不能用于申报书中的模型效果。

### 模型训练、评估与部署

`train.py`

- 读取清单，建立 DataLoader。
- 训练三分类模型或连续转向回归模型。
- 分类默认使用按类别频数加权的交叉熵。
- 验证集以 balanced accuracy 选择最佳模型。
- 保存 `best.pt`、`last.pt` 和逐轮 `history.json`。
- 加 `--use-snn-prior` 时启用 SNN 条件分支。

`predict.py`

- 对单张图片推理。
- 分类模型输出左/直行/右概率和推理耗时。
- 融合模型可同时传入三项 SNN 概率或一个临时硬标签。
- 回归模型输出 CARLA 归一化转向量；角度换算目前只是展示近似，不应当作真实方向盘转角。

`evaluate.py`

- 在指定 train/val/test 集上统一评估。
- 分类输出 accuracy、balanced accuracy、三类 recall、3×3 混淆矩阵和逐图概率。
- 回归输出 MAE 与 RMSE。

`benchmark.py`

- 先进行预热，再重复执行 batch=1 推理。
- 报告模型前向计算的 mean、P50、P95、P99、最小值和最大值。
- 不包含图片读取、预处理、ROS 传输与控制下发，因此不能冒充端到端延迟。

`export_onnx.py`

- 把训练好的 PyTorch 检查点导出为 ONNX。
- 便于后续在 Jetson 上按对应 JetPack/TensorRT 版本部署。
- ONNX 成功不等于 TensorRT 一定兼容，仍需在目标板上实测。

### `src/ann_baseline` 内部模块

`constants.py`：类别顺序、类别数和 ImageNet 归一化常数。

`interfaces.py`：定义 SNN 概率的归一化、硬标签转换、水平翻转和缺失值表示。

`data.py`：读取清单、打开图片、数据增强、水平翻转时同步交换左右标签/SNN 概率，并构造训练张量。

`model.py`：MobileNetV3-Small 图像分支、SNN 小型编码分支和最终分类/回归头。

`metrics.py`：三分类混淆矩阵、准确率、各类召回率、balanced accuracy，以及回归 MAE/RMSE。

`checkpoint.py`：保存与加载模型参数、类别顺序、输入尺寸和验证指标。

`tests/test_smoke.py`：检查 SNN 概率接口、模型输出形状和回归范围。

`ros_msgs/SnnPrior.msg`：建议的 ROS 2 SNN 消息字段。

## 四、为什么不建议简单地让 ANN “等 SNN 输出后再运行”

如果 SNN 的价值是先给出低延迟判断，而 ANN 必须等 SNN 完成后才开始，系统就从并行变成串行，总延迟约为：

```text
T_total = T_SNN + T_ANN + T_fusion
```

这会削弱项目“快慢双通路”的核心亮点。更合适的是：

```text
摄像头/事件流
   ├─> SNN 快通路 ──> 临时风险/方向建议 ─────────┐
   │                                            │
   └─> ANN 图像通路 ──> 场景与精细方向概率 ─────┼─> 仲裁器 ─> 控制命令
                         ▲                      │
                         └── 可选读取 SNN 先验 ──┘
```

SNN 可以先发布暂态建议；ANN 随后给出更完整判断；仲裁器结合两者的置信度、输出年龄、车辆速度和安全规则决定最终动作。ANN 可以使用 SNN 作为条件，但系统不能因此失去 SNN 的快速旁路。

异步多模态网络是可继续学习的方向：[Gehrig 等的 RAM Net](https://arxiv.org/abs/2102.09320)使用内部状态处理不规则到达的事件和图像；CARLA 官方传感器数据也同时带有 `frame` 与仿真 `timestamp`，适合做严格对齐：[CARLA Sensors and data](https://carla.readthedocs.io/en/latest/core_sensors/)。

## 五、四种融合方法及本项目推荐顺序

### 1. 决策级后融合：第一种正式基线

分别得到 ANN 和 SNN 的校准 logits：

```text
z_fusion = z_ann / T_ann + λ · q · z_snn / T_snn
p_fusion = softmax(z_fusion)
```

- `T_ann`、`T_snn` 是在验证集上拟合的温度。
- `q` 是 SNN 可用性/新鲜度/可靠性门控，超时为 0。
- `λ` 只在验证集选择，不能在测试集调。

优点：两套模型可独立训练、容易排错、容易做 SNN-only/ANN-only/fusion 对照。缺点：只融合最终结论，没有充分利用中间特征。

在概率融合前需要校准，因为神经网络 softmax 往往过度自信；温度缩放是简单基线：[Guo 等，On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)。

### 2. 特征拼接：当前代码已经实现

```text
h_img = MobileNet(image)
h_snn = MLP([p_left, p_straight, p_right, mask])
prediction = Head(concat(h_img, h_snn))
```

推荐先冻结或分别训练好两个主干，再只训练融合头；否则小数据下容易让模型完全依赖某一支。训练时应随机把一部分 SNN 输入置为缺失，保证真实系统掉帧时 ANN 仍能工作。

相关的直接案例是 [SNN-ANN Hybrid Networks for Embedded Multimodal Monocular Depth Estimation](https://openreview.net/forum?id=PFHxqiGD8H)：SNN 从事件中提取稀疏表示，再与帧图像 ANN 结合。它的任务不是转向判断，因此只能作为结构参考，不能直接照搬其性能结论。

### 3. FiLM/门控调制：数据足够后再做

让 SNN 条件向量生成每个 ANN 特征通道的缩放与偏移：

```text
γ, β = MLP(snn_prior)
h'_img = γ ⊙ h_img + β
```

相比末端拼接，它允许 SNN 更早地影响图像特征；但调试和过拟合风险更高。方法原型来自 [FiLM](https://ojs.aaai.org/index.php/AAAI/article/view/11671)。

更近的 SNN–ANN 视觉实例可参考 [CVPR 2025 的混合事件目标检测网络](https://openaccess.thecvf.com/content/CVPR2025/papers/Ahmed_Efficient_Event-Based_Object_Detection_A_Hybrid_Neural_Network_with_Spatial_CVPR_2025_paper.pdf)，其中桥接模块把稀疏时空 SNN 表示转换为 ANN 可处理的稠密特征。

### 4. 端到端联合训练：最后阶段

在独立模型和融合头都稳定后，再尝试联合微调：

```text
L = L_fusion + αL_ann_aux + βL_snn_aux + γL_temporal
```

辅助损失可防止某个分支在联合训练中失效。必须同时保留独立分支结果、缺失模态测试、延迟与闭环驾驶实验，否则离线准确率提高不能证明系统更可靠。

## 六、黄祺茵当前只有“左/右”输出时怎样接

不能直接把二分类 `[p_left, p_right]` 变成 `[p_left, 0, p_right]`。那相当于宣称 SNN 认为“直行概率永远为零”，语义是错误的。

推荐把 SNN 输出改成：

```text
q_turn                  = 当前帧是否需要转向/避障
p_left_given_turn       = 需要转向时向左的条件概率
p_right_given_turn      = 需要转向时向右的条件概率
```

再转换：

```text
p_left     = q_turn · p_left_given_turn
p_straight = 1 - q_turn
p_right    = q_turn · p_right_given_turn
```

如果她的网络目前没有 `q_turn`，短期有两个安全选择：

1. SNN 保留二分类语义，作为两个辅助特征输入 ANN，而不冒充完整三分类概率。
2. 暂不启用 SNN 条件分支，先完成 ANN-only 三分类；等 SNN 扩成三分类或增加 `q_turn` 后再正式融合。

还要先确认“左/右”的物理语义：是“障碍物位于左/右”，还是“车辆应该向左/右”。前者的规避方向通常与后者相反，接口名必须写清楚。

## 七、时间同步、缺失和安全仲裁

正式消息至少包含：

```json
{
  "frame_id": 18452,
  "timestamp_ns": 1784971248123456789,
  "left_probability": 0.20,
  "straight_probability": 0.15,
  "right_probability": 0.65,
  "inference_latency_ms": 12.6,
  "model_version": "snn-v2"
}
```

融合前必须检查：

1. `frame_id` 是否对应同一摄像头帧，或时间差是否在允许窗口内。
2. 输出年龄是否超过 TTL；过期结果不能复用成当前帧结果。
3. 三个概率是否有限、非负，归一化后和是否为 1。
4. 模型版本、预处理和类别顺序是否一致。
5. SNN 与 ANN 高置信度冲突时，仲裁器应减速/保持安全动作，而不是武断选一个。

CARLA 调试建议使用同步模式和固定时间步；ROS bridge 的同步模式会等待当前帧预期传感器消息，有助于复现实验：[CARLA ROS bridge 同步说明](https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/)。

## 八、下一阶段“输出转向角度”

现阶段三分类只给粗粒度动作。下一阶段应在采集时保存每帧专家控制 `carla.VehicleControl.steer`，其标注范围按 CARLA 接口使用 `[-1, 1]`，而不是凭图片类别虚构角度。

建议做多任务模型：

```text
共享图像/SNN特征
   ├─ 分类头：left / straight / right
   └─ 回归头：steer ∈ [-1, 1]
```

损失可以从：

```text
L = CrossEntropy(direction) + μ · SmoothL1(steer)
```

开始。三分类头有助于稳定语义，回归头提供连续控制。离线应报告 MAE、RMSE、P95 绝对误差和相邻帧抖动；最终还要在 CARLA 闭环报告路线完成率、碰撞率、越线率、人工接管率和端到端 P50/P95/P99 延迟。

## 九、严格的实验对照

每次只改变一个变量，至少保留：

1. SNN-only。
2. ANN-only。
3. 固定权重后融合。
4. 学习式特征融合。
5. SNN 缺失、延迟和错误概率扰动。
6. 三者在完全独立 route/scenario 上的结果。

假 SNN 只用于第 3、4 项的“程序能否运行”检查，不得进入性能表。正式融合结果必须来自真实 SNN 对同一批、同一帧图像的输出。
