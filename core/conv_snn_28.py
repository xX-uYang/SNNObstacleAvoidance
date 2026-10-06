import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

#接收彩色图片版
class ConvSNN28(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv = nn.Sequential(
            # 输入 1x28x28
            layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 16x28x28
            neuron.LIFNode(tau=2.0, v_threshold=0.5),
            layer.MaxPool2d(2, 2),  # 16x14x14

            layer.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32x14x14
            neuron.LIFNode(tau=2.0, v_threshold=0.5),
            layer.MaxPool2d(2, 2),  # 32x7x7

            layer.Flatten(),
            layer.Linear(32*7*7, 128),
            neuron.LIFNode(tau=2.0, v_threshold=0.5),
            layer.Linear(128, num_classes),
            neuron.LIFNode(tau=2.0, v_threshold=0.5)
        )

    def forward(self, x):
        return self.conv(x)