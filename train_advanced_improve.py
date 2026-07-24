import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from advanced_snn import AdvancedSNN
from dataset_general import CarDatasetGeneral
from spikingjelly.activation_based import functional
from config_real import Config

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 配置区域 =========3
    img_size = Config.IMG_SIZE
    in_channels = Config.IN_CHANNELS
    num_classes = Config.NUM_CLASSES
    batch_size = Config.BATCH_SIZE
    T = Config.T
    epochs = Config.EPOCHS
    lr = Config.LR
    v_threshold = Config.V_THRESHOLD
    tau = Config.TAU
    hidden_channels = Config.HIDDEN_CHANNELS
    data_root = Config.DATA_ROOT
    model_save_path = 'advanced_snn_best.pth'
    # ======================

    dataset = CarDatasetGeneral(data_root, img_size=img_size, in_channels=in_channels)
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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        net.train()  # 设置网络为训练模式
        correct = 0
        total = 0
        total_loss = 0

        for imgs, labels in loader:  # 遍历每个batch
            imgs, labels = imgs.to(device), labels.to(device)
            # 将同一张图像重复T次，构造时间步序列 [T, batch, C, H, W]
            imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)

            out_sum = 0
            for t in range(T):  # 模拟T个时间步
                out_sum += net(imgs_seq[t])  # 累积每个时间步的输出脉冲

            # 用累积的输出计算损失（相当于投票结果）
            loss = criterion(out_sum, labels)

            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            functional.reset_net(net)  # 重置所有神经元的内部状态（膜电位等）

            total_loss += loss.item()
            _, predicted = out_sum.max(1)  # 取累积输出最大值对应的类别
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total
        print(
            f'Epoch {epoch + 1:2d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {optimizer.param_groups[0]["lr"]:.5f}')

        # 学习率调度
        scheduler.step()

        # 早停和保存最佳模型
        if acc > best_acc:
            best_acc = acc
            no_improve = 0
            torch.save(net.state_dict(), model_save_path)
            print(f'  -> Best model saved (acc={best_acc:.2f}%)')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    print(f'Training finished. Best accuracy: {best_acc:.2f}%')
    print(f'Best model saved as {model_save_path}')

if __name__ == '__main__':
    train()