import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from advanced_snn import AdvancedSNN
from dataset_general import CarDatasetGeneral
from spikingjelly.activation_based import functional
from config_small import Config

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 从配置读取参数
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
    model_save_path = 'small_model.pth'

    # 加载数据集
    dataset = CarDatasetGeneral(data_root, img_size=img_size, in_channels=in_channels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建网络
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

    best_acc = 0.0
    for epoch in range(epochs):
        net.train()
        correct = 0
        total = 0
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T, B, C, H, W]

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

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), model_save_path)
            print(f'  -> 保存最佳模型，准确率 {best_acc:.2f}%')
        if acc >= 100.0:
            print('训练准确率已达100%，停止训练')
            break

    print(f'训练结束，最佳准确率: {best_acc:.2f}%')

if __name__ == '__main__':
    train()