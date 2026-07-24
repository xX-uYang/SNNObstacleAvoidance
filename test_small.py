import torch
import os
from PIL import Image
from torchvision import transforms
from advanced_snn import AdvancedSNN
from config_small import Config

device = torch.device('cpu')
# 加载模型
net = AdvancedSNN(
    num_classes=Config.NUM_CLASSES,
    in_channels=Config.IN_CHANNELS,
    img_h=Config.IMG_SIZE[0],
    img_w=Config.IMG_SIZE[1],
    hidden_channels=Config.HIDDEN_CHANNELS,
    v_threshold=Config.V_THRESHOLD,
    tau=Config.TAU
)
net.load_state_dict(torch.load('small_model.pth', map_location=device))
net.eval()

# 预处理（必须与训练一致）
transform_list = [transforms.Resize(Config.IMG_SIZE)]
if Config.IN_CHANNELS == 1:
    transform_list.append(transforms.Grayscale())
transform_list.append(transforms.ToTensor())
transform_list.append(transforms.Normalize(mean=[0.5]*Config.IN_CHANNELS, std=[0.5]*Config.IN_CHANNELS))
transform = transforms.Compose(transform_list)

# 遍历所有图片
root = Config.DATA_ROOT
class_names = ['left', 'right', 'straight']
for cls in class_names:
    folder = os.path.join(root, cls)
    if not os.path.exists(folder):
        continue
    print(f'\n--- {cls} 文件夹 ---')
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = Image.open(path)
        if Config.IN_CHANNELS == 3 and img.mode != 'RGB':
            img = img.convert('RGB')
        input_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]

        # 时间步推理
        T = Config.T
        input_seq = input_tensor.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        with torch.no_grad():
            out_sum = 0
            for t in range(T):
                out_sum += net(input_seq[t])
            _, pred = out_sum.max(1)

        true_label = class_names.index(cls)
        pred_label = class_names[pred.item()]
        correct = (pred.item() == true_label)
        print(f'{fname}: 真实={cls}, 预测={pred_label}  {"✓" if correct else "✗"}')