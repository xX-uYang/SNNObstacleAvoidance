import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# test_inference.py
import torch
from core.flexible_conv_snn import FlexibleConvSNN
from PIL import Image
from torchvision import transforms
import argparse

def predict(image_path, model_path, img_size=(64,64), in_channels=3, T=16):
    device = torch.device('cpu')
    # 构建网络（必须和训练时一致）
    net = FlexibleConvSNN(num_classes=3, in_channels=in_channels, img_h=img_size[0], img_w=img_size[1], time_step=T)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    # 预处理（必须和训练时一致）
    transform_list = [transforms.Resize(img_size)]
    if in_channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.append(transforms.ToTensor())
    if in_channels == 1:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    else:
        transform_list.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
    transform = transforms.Compose(transform_list)

    img = Image.open(image_path)
    if in_channels == 3 and img.mode != 'RGB':
        img = img.convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]

    # 时间步推理
    img_seq = img_tensor.unsqueeze(0).repeat(T, 1, 1, 1, 1)
    with torch.no_grad():
        out_sum = 0
        for t in range(T):
            out_sum += net(img_seq[t])
        _, pred = out_sum.max(1)

    classes = ['left', 'right', 'straight']
    print(f'预测结果: {classes[pred.item()]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--model', type=str, default='flexible_car_snn.pth', help='模型文件')
    parser.add_argument('--size', type=int, nargs=2, default=[64,64], help='图片尺寸')
    parser.add_argument('--channels', type=int, default=3, help='输入通道数')
    parser.add_argument('--T', type=int, default=16, help='时间步')
    args = parser.parse_args()

    predict(args.image, args.model, img_size=tuple(args.size), in_channels=args.channels, T=args.T)