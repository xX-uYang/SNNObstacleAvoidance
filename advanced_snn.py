import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer

class AdvancedSNN(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, img_h=64, img_w=64,
                 hidden_channels=[32, 64, 128], v_threshold=0.5, tau=2.0):
        super().__init__()
        layers = []
        in_ch = in_channels
        h, w = img_h, img_w
        for out_ch in hidden_channels:
            # 两个卷积层（可自行增减）
            layers.append(layer.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(neuron.LIFNode(tau=tau, v_threshold=v_threshold))
            layers.append(layer.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
            layers.append(neuron.LIFNode(tau=tau, v_threshold=v_threshold))
            layers.append(layer.MaxPool2d(2, 2))
            in_ch = out_ch
            h, w = h//2, w//2

        self.features = nn.Sequential(*layers)

        # 自动计算全连接输入维度
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