# flexible_conv_snn.py
import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer

class FlexibleConvSNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, img_h=64, img_w=64, time_step=16, v_threshold=0.5, tau=2.0):
        super().__init__()
        self.time_step = time_step  # 保留时间步信息，方便后续
        # 计算经过三层卷积+池化后的特征图尺寸
        def conv_output_size(size, kernel=3, stride=1, padding=1, pool=2):
            size = (size + 2 * padding - kernel) // stride + 1
            size = size // pool
            return size
        h = conv_output_size(img_h)
        w = conv_output_size(img_w)
        h = conv_output_size(h)
        w = conv_output_size(w)
        h = conv_output_size(h)
        w = conv_output_size(w)
        self.feature_size = 128 * h * w  # 第三层卷积输出通道为128

        self.conv = nn.Sequential(
            layer.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Linear(self.feature_size, 256),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold),
            layer.Linear(256, num_classes),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold)
        )

    def forward(self, x):
        return self.conv(x)