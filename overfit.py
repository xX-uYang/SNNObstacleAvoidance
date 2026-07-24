# overfit_test.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from advanced_snn import AdvancedSNN
from dataset_general import CarDatasetGeneral
from spikingjelly.activation_based import functional

data_root = "D:/SNN/260417_real_data/train"   # 修改为实际路径
img_size = (64, 64)
in_channels = 3
batch_size = 4
T = 16
lr = 0.01
epochs = 100

dataset = CarDatasetGeneral(data_root, img_size=img_size, in_channels=in_channels)
# 取前16个样本（确保各类都有）
indices = list(range(16))
subset = Subset(dataset, indices)
loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

net = AdvancedSNN(num_classes=3, in_channels=in_channels, img_h=img_size[0], img_w=img_size[1])
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    net.train()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        out_sum = 0
        for t in range(T):
            out_sum += net(imgs_seq[t])
        loss = criterion(out_sum, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        functional.reset_net(net)
        _, pred = out_sum.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    acc = 100. * correct / total
    print(f"Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")
    if acc == 100.0:
        print("过拟合成功！")
        break