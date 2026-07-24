import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import time
from core.advanced_snn import AdvancedSNN
from core.config import Config

def speed_test():
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

    T = Config.T
    dummy_input = torch.randn(1, Config.IN_CHANNELS, Config.IMG_SIZE[0], Config.IMG_SIZE[1])
    dummy_seq = dummy_input.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    # 预热
    for _ in range(10):
        out_sum = 0
        for t in range(T):
            out_sum += net(dummy_seq[t])
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # 正式测试
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        out_sum = 0
        for t in range(T):
            out_sum += net(dummy_seq[t])
    end = time.time()
    avg_time = (end - start) / iterations
    fps = 1.0 / avg_time
    print(f'Average inference time per frame (including {T} time steps): {avg_time*1000:.2f} ms')
    print(f'FPS: {fps:.2f}')

if __name__ == '__main__':
    speed_test()