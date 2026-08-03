# SNN–ANN 接口约定 v0.2

## 类别与物理语义

- `left = 0`：车辆当前应向左转/向左避让。
- `straight = 1`：车辆当前应保持直行。
- `right = 2`：车辆当前应向右转/向右避让。
- 下一阶段统一使用 CARLA `steer ∈ [-1,1]`：负为左、零为直行、正为右。

如果 SNN 判断的是“障碍物位于左/右”，不能直接填入“车辆应左/右转”的字段；二者的物理语义不同。

## 单帧消息

```json
{
  "frame_id": 18452,
  "timestamp_ns": 1784971248123456789,
  "left_probability": 0.15,
  "straight_probability": 0.20,
  "right_probability": 0.65,
  "inference_latency_ms": 12.6,
  "model_version": "snn-v2"
}
```

三项概率必须有限、非负；进入 ANN 前会归一化为和等于 1。正式融合应提供概率或可校准 logits，而不是只提供硬标签。

## ANN 条件输入

```text
[left_probability, straight_probability, right_probability, available_mask]
```

- 同步 SNN 输出可用时：`available_mask=1`。
- SNN 超时、丢帧或尚未接入时：`[1/3, 1/3, 1/3, 0]`。
- 不允许把上一帧 SNN 结果冒充当前帧；按 `frame_id` 或时间戳同步并检查 TTL。

## 二分类 SNN 的临时处理

当前 SNN 如果只有“向左/向右”二分类，不能直接令 `straight_probability=0`。推荐增加“是否需要转向”的概率：

```text
p_left     = q_turn * p_left_given_turn
p_straight = 1 - q_turn
p_right    = q_turn * p_right_given_turn
```

没有 `q_turn` 时，二分类结果应保持为独立辅助特征，或暂不启用正式三分类融合。

## ROS 2 消息

仓库提供 `ros_msgs/SnnPrior.msg`：

```text
std_msgs/Header header
uint64 frame_id
float32 left_probability
float32 straight_probability
float32 right_probability
float32 inference_latency_ms
string model_version
```

建议话题：

- 摄像头：`/carla/ego_vehicle/rgb_front/image`
- SNN：`/snn/prior`
- ANN：`/ann/direction`，后续升级为 `/ann/steering`
- 仲裁结果：`/fusion/control_proposal`

## 联合训练约束

1. 真值来自 CARLA 专家控制、路线规划器或人工复核，不来自 SNN 预测。
2. 训练、验证、测试按 episode/route/scenario/time group 划分。
3. 每张图只属于一个划分，连续帧不能跨划分。
4. 同时报告 SNN-only、ANN-only、fusion。
5. 记录模型版本、预处理、类别顺序、推理硬件、批量大小和端到端延迟。
6. 假 SNN 概率只能做接口测试，不能作为性能证据。

