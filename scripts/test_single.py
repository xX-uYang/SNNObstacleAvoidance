import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from core.advanced_snn import AdvancedSNN
from PIL import Image
from torchvision import transforms

def test_single(image_path):
    # 参数必须与训练时一致
    img_size = (64, 64)
    T = 32
    device = torch.device('cpu')

    net = AdvancedSNN(num_classes=3, in_channels=3, img_h=64, img_w=64,
                      hidden_channels=[16, 32, 64])
    net.load_state_dict(torch.load('advanced_snn_best.pth', map_location=device))
    net.eval()

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # [1,3,64,64]
    input_seq = input_tensor.unsqueeze(0).repeat(T, 1, 1, 1, 1)

    with torch.no_grad():
        out_sum = 0
        for t in range(T):
            out_sum += net(input_seq[t])
        _, pred = out_sum.max(1)
    labels = ['left', 'right', 'straight']
    print(f'预测结果: {labels[pred.item()]}')

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        test_single(sys.argv[1])
    else:
        print("用法: python test_single.py 图片路径")