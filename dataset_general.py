#通用接收新数据（图片形式）
# dataset_general.py
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CarDatasetGeneral(Dataset):
    def __init__(self, root_dir, img_size=(64,64), in_channels=3):
        """
        root_dir: 数据文件夹，结构应为 root_dir/left, root_dir/right, root_dir/straight
        img_size: 输出图片大小 (h, w)
        in_channels: 1 或 3，决定是否转为灰度
        """
        self.images = []
        self.labels = []
        self.class_to_idx = {'left': 0, 'right': 1, 'straight': 2}
        self.img_size = img_size
        self.in_channels = in_channels

        for class_name in ['left', 'right', 'straight']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"加载了 {len(self.images)} 张图片，尺寸将被缩放到 {img_size}，通道数 = {in_channels}")

        # 基础预处理：缩放、转Tensor
        transform_list = [transforms.Resize(img_size)]
        if in_channels == 1:
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        transform_list.append(transforms.ToTensor())
        if in_channels == 1:
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        else:
            transform_list.append(transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]))
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        # 如果 in_channels=3 但图片可能是灰度，用 convert('RGB') 强制三通道；如果 in_channels=1，则后续 Grayscale 会处理
        if self.in_channels == 3 and img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.labels[idx]
        img = self.transform(img)
        return img, label