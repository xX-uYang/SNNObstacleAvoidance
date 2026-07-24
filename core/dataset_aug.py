import random
from torchvision import transforms
from PIL import Image
from dataset_general import CarDatasetGeneral

class CarDatasetAug(CarDatasetGeneral):
    def __init__(self, root_dir, img_size=(64,64), in_channels=3):
        super().__init__(root_dir, img_size, in_channels)
        # 基础transform（不包括ToTensor和Normalize，因为需要在增强后应用）
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*in_channels, std=[0.5]*in_channels)
        ])

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        label = self.labels[idx]

        # 随机水平翻转（同时翻转标签）
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if label == 0:      # left
                label = 1
            elif label == 1:    # right
                label = 0
            # straight 不变

        # 随机色彩抖动（仅当in_channels=3时）
        if self.in_channels == 3 and random.random() < 0.5:
            img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)(img)

        # 随机旋转（小角度）
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = img.rotate(angle)

        img = self.base_transform(img)
        return img, label