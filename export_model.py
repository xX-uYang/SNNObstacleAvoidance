import torch
from advanced_snn import AdvancedSNN
from config import Config

def export():
    device = torch.device('cpu')
    net = AdvancedSNN(
        num_classes=Config.NUM_CLASSES,
        in_channels=Config.IN_CHANNELS,
        img_h=Config.IMG_SIZE[0],
        img_w=Config.IMG_SIZE[1],
        hidden_channels=Config.HIDDEN_CHANNELS,
        v_threshold=Config.V_THRESHOLD,
        tau=Config.TAU
    )
    net.load_state_dict(torch.load('advanced_snn_best.pth', map_location=device))
    net.eval()

    # 转换为TorchScript
    example_input = torch.randn(1, Config.IN_CHANNELS, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    traced_net = torch.jit.trace(net, example_input)
    traced_net.save('advanced_snn_best.pt')
    print("Model exported to advanced_snn_best.pt")

if __name__ == '__main__':
    export()