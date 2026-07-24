# train_general.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flexible_conv_snn import FlexibleConvSNN
from dataset_general import CarDatasetGeneral
from spikingjelly.activation_based import functional

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ====== 配置区域 ======
    img_size = (28, 28)  # 图片尺寸（与数据一致）
    in_channels = 1  # 灰度图，通道数为1
    num_classes = 3
    batch_size = 32
    T = 16  # 时间步
    epochs = 30
    lr = 0.01
    v_threshold = 0.5
    tau = 2.0
    data_root = 'my_car_data_28/train'  # 你的数据路径
    model_save_path = 'flexible_snn_28.pth'
    # ======================

    # 数据集
    train_dataset = CarDatasetGeneral(data_root, img_size=img_size, in_channels=in_channels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 网络
    net = FlexibleConvSNN(
        num_classes=num_classes,
        in_channels=in_channels,
        img_h=img_size[0],
        img_w=img_size[1],
        time_step=T,
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

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # 时间步展开
            imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)

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
            print("提前停止")
            break

    torch.save(net.state_dict(), model_save_path)
    print(f"模型已保存为 {model_save_path}")

if __name__ == '__main__':
    train()