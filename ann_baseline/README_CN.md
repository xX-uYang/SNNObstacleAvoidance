# ANN 三分类与 SNN 融合原型

当前版本已经适配 `E:\Intuition\carla_data`：

- 图像输入；
- 输出 `left / straight / right` 三类概率；
- 可选读取三类 SNN 概率；
- SNN 缺失时通过 `available_mask` 退回图像判断；
- 下一阶段可切换为 CARLA `steer ∈ [-1,1]` 回归；
- 可评估、单图推理与导出 ONNX。

详细原理、全部脚本解释和融合方案见：

- [脚本说明与SNN-ANN融合指南](docs/脚本说明与SNN-ANN融合指南.md)
- [SNN-ANN接口约定](docs/SNN-ANN接口约定.md)

## 已准备的数据

```text
data/
  carla_manifest.csv
  carla_manifest_with_fake_snn.csv
```

真实数据清单使用时间组划分，避免连续相邻帧被随机拆到不同集合。假 SNN 清单由真实标签派生，只能调试接口，不能报告其中的性能。

## 重新生成清单

```powershell
python prepare_carla_manifest.py `
  --dataset-root E:\Intuition\carla_data `
  --output E:\Intuition\code\ann_baseline\data\carla_manifest.csv

python generate_fake_snn_outputs.py `
  --manifest E:\Intuition\code\ann_baseline\data\carla_manifest.csv `
  --output E:\Intuition\code\ann_baseline\data\carla_manifest_with_fake_snn.csv
```

## ANN-only 三分类训练

```powershell
python train.py `
  --manifest data\carla_manifest.csv `
  --output-dir runs\ann_image_v1 `
  --pretrained `
  --epochs 20
```

## 假 SNN 接口调试

```powershell
python train.py `
  --manifest data\carla_manifest_with_fake_snn.csv `
  --output-dir runs\fusion_plumbing_only `
  --use-snn-prior `
  --pretrained `
  --epochs 3
```

该命令仅检查融合链路，不产生有效的融合实验结论。

## 评估与单图预测

```powershell
python evaluate.py `
  --checkpoint runs\ann_image_v1\best.pt `
  --manifest data\carla_manifest.csv `
  --split test `
  --output-dir runs\ann_image_v1_test

python predict.py `
  --checkpoint runs\ann_image_v1\best.pt `
  --image E:\Intuition\carla_data\left\某张图片.png

python benchmark.py `
  --checkpoint runs\ann_image_v1\best.pt `
  --image E:\Intuition\carla_data\left\某张图片.png `
  --device cuda
```

真实 SNN 接入后三类概率示例：

```powershell
python predict.py `
  --checkpoint runs\ann_fusion_v1\best.pt `
  --image demo.png `
  --snn-left 0.15 `
  --snn-straight 0.20 `
  --snn-right 0.65
```

## 当前环境与第一轮结果

已在 `D:\python3.9\Intuition` 安装 CUDA 版 PyTorch，并使用 RTX 4060 完成 20 轮 ANN-only 训练。模型和结果保存在 `E:\Intuition\runs`。

- 测试 accuracy：85.23%
- 测试 balanced accuracy：84.96%
- 左/直行/右召回率：87.10% / 77.78% / 90.00%
- 预热后模型前向 P50/P95：3.64 ms / 4.89 ms

完整结果和限制见 [ANN第一轮训练结果](../../runs/ANN第一轮训练结果.md)。假 SNN 融合也已跑通，但其数据由标签派生，只能证明代码链路正常，不能报告为融合性能。
