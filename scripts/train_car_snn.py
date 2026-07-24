import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

from spikingjelly.activation_based import functional, neuron, layer


# -------------------- 1. 定义网络结构 --------------------
class CarIntuitionSNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入图片是 128x128 的灰度图
        self.net = nn.Sequential(
            layer.Flatten(),  # 把128x128拉平成16384
            layer.Linear(128 * 128, 128),  # 第一层：16384 -> 128
            neuron.LIFNode(tau=2.0),  # 脉冲神经元
            layer.Linear(128, 3),  # 输出层：3类（左、直、右）
            neuron.LIFNode(tau=2.0)
        )

    def forward(self, x):
        return self.net(x)


# -------------------- 2. 自定义数据集类 --------------------
class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: 数据文件夹，结构应为：
            root_dir/
                left/    # 左转图片
                right/   # 右转图片
                straight/ # 直行图片
        """
        self.images = []
        self.labels = []
        self.transform = transform

        # 类别到数字的映射
        self.class_to_idx = {'left': 0, 'right': 1, 'straight': 2}

        # 遍历每个类别文件夹
        for class_name in ['left', 'right', 'straight']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告：文件夹 {class_dir} 不存在")
                continue

            # 获取该文件夹下所有图片
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"加载了 {len(self.images)} 张图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # 转成灰度图

        # 应用变换（调整大小、转成tensor）
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# -------------------- 3. 训练函数 --------------------
def train():
    # 设置参数
    device = torch.device('cpu')
    batch_size = 32
    epochs = 10
    lr = 0.001

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 统一大小
        transforms.ToTensor(),  # 转成tensor (0-1)
        transforms.Normalize((0.5,), (0.5,))  # 归一化到-1到1
    ])

    # 加载数据（请确认这个路径正确）
    train_dataset = CarDataset(
        root_dir='D:/SNN/my_car_data/train',  # 这里改成你的训练数据路径
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # 初始化网络
    net = CarIntuitionSNN().to(device)
    print(net)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # 训练循环
    print("开始训练...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = net(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 重置脉冲神经元的内部状态（重要！）
            functional.reset_net(net)

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 10 == 9:
                print(
                    f'Epoch {epoch + 1}, Batch {i + 1}: loss={running_loss / 10:.3f}, acc={100. * correct / total:.2f}%')
                running_loss = 0.0

        print(f'Epoch {epoch + 1} 完成，准确率: {100. * correct / total:.2f}%')

    # 保存模型
    torch.save(net.state_dict(), 'car_intuition_snn.pth')
    print("模型已保存为 car_intuition_snn.pth")


# -------------------- 4. 运行训练 --------------------
if __name__ == '__main__':
    train()