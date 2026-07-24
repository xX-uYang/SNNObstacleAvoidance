# config.py
class Config:
    # 数据
    DATA_ROOT = 'my_car_data_28/train'
    IMG_SIZE = (28, 28)
    IN_CHANNELS = 1
    NUM_CLASSES = 3

    # 训练
    BATCH_SIZE = 32
    T = 32
    EPOCHS = 30
    LR = 0.01
    V_THRESHOLD = 0.5
    TAU = 2.0

    # 模型
    HIDDEN_CHANNELS = [16, 32]   # 适用于AdvancedSNN

    # 增强
    USE_AUGMENTATION = False