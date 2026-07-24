import torch
from conv_snn_28 import ConvSNN28
from PIL import Image
from torchvision import transforms

def predict(image_path, model_path, T=16):
    device = torch.device('cpu')
    # 加载模型（必须使用训练时的网络类）
    net = ConvSNN28(num_classes=3)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(image_path).convert('RGB')

    img_tensor = transform(img).unsqueeze(0)   # [1,1,28,28]

    # 时间步推理
    img_seq = img_tensor.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [T,1,1,28,28]
    with torch.no_grad():
        out_sum = 0
        for t in range(T):
            out_sum += net(img_seq[t])
        _, pred = out_sum.max(1)

    classes = ['left', 'right', 'straight']
    print(f'预测结果: {classes[pred.item()]}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='图片路径')
    parser.add_argument('--model', type=str, default='car_snn_color_test.pth', help='模型文件')
    parser.add_argument('--T', type=int, default=16, help='时间步')
    args = parser.parse_args()
    predict(args.image, args.model, T=args.T)