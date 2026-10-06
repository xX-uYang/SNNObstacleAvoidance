import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# train_conv_snn_color.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from core.conv_snn_28 import ConvSNN28          # 导入已经修改为3通道的网络
from core.dataset_color import CarDatasetColor   # 导入刚刚创建的数据集类
from spikingjelly.activation_based import functional

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 超参数
    img_size = (28, 28)
    batch_size = 64
    T = 32                     # 时间步
    epochs = 30
    lr = 0.01                  # 学习率
    num_classes = 3

    # 数据加载
    train_dataset = CarDatasetColor('my_car_data_28/train', img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 初始化网络（因为ConvSNN28内部第一层已经是3通道，所以直接使用）
    net = ConvSNN28(num_classes=num_classes).to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        correct = 0
        total = 0
        total_loss = 0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 重复时间步
            imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T, B, C, H, W]

            out_sum = 0
            for t in range(T):
                out = net(imgs_seq[t])
                out_sum += out

            loss = criterion(out_sum, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            total_loss += loss.item()
            _, predicted = out_sum.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        print(f'Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')

        if acc > 95:
            print("准确率超过95%，停止训练。")
            break

    torch.save(net.state_dict(), 'car_snn_color_test.pth')
    print("模型已保存为 car_snn_color_test.pth")

if __name__ == '__main__':
    train()