import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# evaluate_single.py
import torch
from core.dataset_general import CarDatasetGeneral
from core.advanced_snn import AdvancedSNN
from spikingjelly.activation_based import functional

def evaluate():
    device = torch.device('cpu')
    img_size = (64, 64)
    in_channels = 3
    T = 32
    test_root = 'D:/SNN/260427_real_data_split/test'   # 请修改为你的测试集路径

    # 加载数据集
    dataset = CarDatasetGeneral(test_root, img_size=img_size, in_channels=in_channels)
    print(f"测试集总数: {len(dataset)}")

    # 创建网络（必须与训练时结构一致）
    net = AdvancedSNN(
        num_classes=3,
        in_channels=3,
        img_h=64,
        img_w=64,
        hidden_channels=[16, 32, 64],   # 与训练时保持一致
        v_threshold=0.5,
        tau=2.0
    )
    net.load_state_dict(torch.load('advanced_snn_best.pth', map_location=device))
    net.eval()

    correct = 0
    total = len(dataset)

    with torch.no_grad():
        for idx in range(total):
            img, label = dataset[idx]
            # 添加 batch 维度
            img = img.unsqueeze(0)  # [1, 3, 64, 64]
            # 重复时间步
            img_seq = img.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T, 1, 3, 64, 64]

            out_sum = 0
            for t in range(T):
                out_sum += net(img_seq[t])

            _, pred = out_sum.max(1)
            if pred.item() == label:
                correct += 1

            # 重置网络内部状态，避免跨样本影响
            functional.reset_net(net)

    acc = 100.0 * correct / total
    print(f'测试集准确率: {acc:.2f}% ({correct}/{total})')

if __name__ == '__main__':
    evaluate()