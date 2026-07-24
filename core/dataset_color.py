# dataset_color.py
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CarDatasetColor(Dataset):
    def __init__(self, root_dir, img_size=(28,28)):
        """
        root_dir: 数据文件夹，结构应为 root_dir/left, root_dir/right, root_dir/straight
        img_size: 输出图片大小 (h, w)
        """
        self.images = []
        self.labels = []
        self.class_to_idx = {'left': 0, 'right': 1, 'straight': 2}
        self.img_size = img_size

        # 遍历每个类别文件夹，收集图片路径
        for class_name in ['left', 'right', 'straight']:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(self.class_to_idx[class_name])

        print(f"加载了 {len(self.images)} 张图片，用于彩色训练测试。")

        # 定义预处理：缩放、转为Tensor、归一化（三通道）
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),  # 此时输出为 3 x H x W，值范围[0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        # 关键：用 convert('RGB') 将灰度图转为三通道（每个通道值相同）
        # 用现有灰度图进行测试
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        img = self.transform(img)
        return img, label