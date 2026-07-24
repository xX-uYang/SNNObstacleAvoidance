import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from core.advanced_snn import AdvancedSNN
from core.dataset_aug import CarDatasetAug   # 导入增强版
from spikingjelly.activation_based import functional

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ====== 配置区域 ======
    img_size = (28, 28)
    in_channels = 1
    num_classes = 3
    batch_size = 32
    T = 16
    epochs = 30
    lr = 0.01
    v_threshold = 0.5
    tau = 2.0
    hidden_channels = [16, 32]
    data_root = 'my_car_data_28/train'
    model_save_path = 'advanced_snn_aug_28.pth'
    # ======================

    dataset = CarDatasetAug(data_root, img_size=img_size, in_channels=in_channels)  # 使用增强数据集
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    net = AdvancedSNN(
        num_classes=num_classes,
        in_channels=in_channels,
        img_h=img_size[0],
        img_w=img_size[1],
        hidden_channels=hidden_channels,
        v_threshold=v_threshold,
        tau=tau
    ).to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()
        correct = 0
        total = 0
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)

            out_sum = 0
            for t in range(T):
                out_sum += net(imgs_seq[t])

            loss = criterion(out_sum, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            total_loss += loss.item()
            _, predicted = out_sum.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total
        print(f'Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')

        if acc > 95:
            print("准确率超过95%，停止训练")
            break

    torch.save(net.state_dict(), model_save_path)
    print(f"模型已保存为 {model_save_path}")

if __name__ == '__main__':
    train()