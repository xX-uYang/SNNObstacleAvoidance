import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from spikingjelly.activation_based import functional, neuron, layer

# -------------------- 1. 数据集类 --------------------
class CarDatasetGeneral(Dataset):
    def __init__(self, root_dir, img_size=(64, 64), in_channels=3):
        self.images = []
        self.labels = []
        self.class_to_idx = {'left': 0, 'right': 1, 'straight': 2}
        self.img_size = img_size
        self.in_channels = in_channels

        for class_name in ['left', 'right', 'straight']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"加载了 {len(self.images)} 张图片，尺寸将被缩放到 {img_size}，通道数 = {in_channels}")

        # 预处理：缩放、转Tensor、归一化
        transform_list = [transforms.Resize(img_size)]
        if in_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.ToTensor())
        if in_channels == 1:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.in_channels == 3 and img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.labels[idx]
        img = self.transform(img)
        return img, label

# -------------------- 2. 网络定义 --------------------
class AdvancedSNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, img_h=64, img_w=64,
                 hidden_channels=[16, 32, 64], v_threshold=0.5, tau=2.0):
        super().__init__()
        layers = []
        in_ch = in_channels
        h, w = img_h, img_w
        for out_ch in hidden_channels:
            layers.append(layer.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(neuron.LIFNode(tau=tau, v_threshold=v_threshold))
            layers.append(layer.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            layers.append(neuron.LIFNode(tau=tau, v_threshold=v_threshold))
            layers.append(layer.MaxPool2d(2, 2))
            in_ch = out_ch
            h, w = h // 2, w // 2

        self.features = nn.Sequential(*layers)
        self.flatten = layer.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            dummy = self.features(dummy)
            fc_in = dummy.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            layer.Linear(fc_in, 256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.Linear(256, num_classes),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

# -------------------- 3. 训练函数 --------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ========== 配置参数（请根据实际情况修改） ==========
    data_root = "D:/SNN/260417_real_data/train"    # 修改为你的数据集路径
    img_size = (64, 64)
    in_channels = 3
    num_classes = 3
    batch_size = 32
    T = 32                       # 时间步（增加，让SNN有更多时间累积信息）
    epochs = 100
    lr = 0.005                   # 学习率（比原来稍低）
    v_threshold = 0.5
    tau = 2.0
    hidden_channels = [16, 32, 64]   # 网络容量（适当增大）
    model_save_path = 'advanced_snn_best.pth'
    # =================================================

    # 加载数据集
    dataset = CarDatasetGeneral(data_root, img_size=img_size, in_channels=in_channels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    patience = 10
    no_improve = 0

    for epoch in range(epochs):
        net.train()
        correct = 0
        total = 0
        total_loss = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            # 构造时间步序列
            imgs_seq = imgs.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T, batch, C, H, W]

            out_sum = 0
            for t in range(T):
                out_sum += net(imgs_seq[t])

            loss = criterion(out_sum, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(net)   # 重置神经元状态

            total_loss += loss.item()
            _, predicted = out_sum.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(loader)
        acc = 100. * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.5f}')

        scheduler.step()

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