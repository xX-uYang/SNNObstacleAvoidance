import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# train_car_snn_28x28_conv.py (你可以重命名)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

from spikingjelly.activation_based import functional, neuron, layer

# ----- 导入新网络（从 conv_snn_28.py 导入）-----
from core.conv_snn_28 import ConvSNN28

# ----- 数据集类（保持不变）-----
class CarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {'left': 0, 'right': 1, 'straight': 2}

        for class_name in ['left', 'right', 'straight']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        print(f"加载了 {len(self.images)} 张28x28图片")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ----- 训练函数（修改网络实例化部分）-----
def train():
    device = torch.device('cpu')
    batch_size = 64
    epochs = 50
    lr = 0.01
    T = 32

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CarDataset('my_car_data_28/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 关键修改：使用 ConvSNN28 替换 SimpleSNN
    net = ConvSNN28(num_classes=3).to(device)
    print(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        correct = 0
        total = 0
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)

            img_seq = img.unsqueeze(0).repeat(T, 1, 1, 1, 1)

            out_sum = 0
            for t in range(T):
                out = net(img_seq[t])
                out_sum += out

            loss = criterion(out_sum, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            _, predicted = out_sum.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

        acc = 100. * correct / total
        print(f'Epoch {epoch+1:2d} 准确率: {acc:.2f}%')
        if acc > 95:
            print("达到95%准确率，停止训练")
            break

    torch.save(net.state_dict(), 'car_snn_28x28_conv.pth')
    print("模型已保存")

if __name__ == '__main__':
    train()