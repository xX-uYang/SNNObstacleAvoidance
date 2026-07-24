# config_small.py
class Config:
    # 数据路径（改成你存放图片的文件夹路径）
    DATA_ROOT = 'D:/SNN/test_small_data/train'
    IMG_SIZE = (64, 64)          # 训练时缩放到64x64
    IN_CHANNELS = 3              # 因为是RGB，所以是3
    NUM_CLASSES = 3

    # 训练参数
    BATCH_SIZE = 4                # 批次小一点，因为总数据只有224张，但left只有3张
    T = 8                         # 时间步减少，加速训练
    EPOCHS = 30                   # 因为数据少，可以多跑几轮观察过拟合
    LR = 0.01
    V_THRESHOLD = 0.5
    TAU = 2.0

    # 模型（减小网络宽度，防止过拟合太快）
    HIDDEN_CHANNELS = [8, 16]     # 比原来[16,32]小

    # 数据增强（先关掉，用原始数据训练）
    USE_AUGMENTATION = False