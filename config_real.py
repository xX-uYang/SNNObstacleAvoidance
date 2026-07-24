# config_real.py
from distributed.protocol import torch


class Config:
    DATA_ROOT = 'D:/SNN/260417_real_data/train'   # 改为你的实际路径
    IMG_SIZE = (64, 64)                    # 可尝试 64x64 或 128x128
    IN_CHANNELS = 3                        # 彩色图
    NUM_CLASSES = 3

    BATCH_SIZE = 32                        # 显存允许可调大
    T = 32                                # 时间步
    EPOCHS = 100
    LR = 0.005
    V_THRESHOLD = 0.5
    TAU = 2.0

    HIDDEN_CHANNELS = [32, 64]             # 可以适当增大宽度（原来小样本是[8,16]）
    USE_AUGMENTATION = False               # 先不加增强，看基线

    # 学习率调度（如果已有，保持；否则添加）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)