import os
from PIL import Image
from torchvision import transforms
from dataset_general import CarDatasetGeneral

# 配置（与训练时一致）
data_root = "D:/SNN/260417_real_data/train"   # 例如 D:/SNN/real_data/train
img_size = (64, 64)
in_channels = 3

dataset = CarDatasetGeneral(data_root, img_size=img_size, in_channels=in_channels)
print(f"总样本数: {len(dataset)}")

# 统计各类别数量
from collections import Counter
labels = [dataset[i][1] for i in range(len(dataset))]
print("类别分布:", Counter(labels))

# 查看第一张图的形状和值范围
img, label = dataset[0]
print(f"图像shape: {img.shape}, 值域: [{img.min():.2f}, {img.max():.2f}], 标签: {label}")