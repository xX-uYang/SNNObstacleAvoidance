import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os

# 复用之前的数据集类
from train_car_snn import CarDataset

print("=" * 50)
print("用普通神经网络测试数据")
print("=" * 50)

# 设置参数
device = torch.device('cpu')
batch_size = 32
epochs = 20
lr = 0.001

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据
dataset = CarDataset('D:/SNN/my_car_data/train', transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 定义一个简单的ANN
class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)


# 初始化网络
ann = SimpleANN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(ann.parameters(), lr=lr)

print("开始训练ANN...")
print("如果loss下降，说明数据本身有规律；如果loss不变，说明数据生成有问题")

# 训练
for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = ann(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    print(f'Epoch {epoch + 1}: loss={total_loss / len(loader):.4f}, acc={100. * correct / total:.2f}%')

print("\n" + "=" * 50)
print("如果准确率 > 80%，说明数据没问题，是SNN训练的问题")
print("如果准确率还在30-40%，说明数据本身就有问题")
print("=" * 50)