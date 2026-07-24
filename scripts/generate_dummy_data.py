import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import os

# 创建文件夹
os.makedirs('my_car_data_28/train/left', exist_ok=True)
os.makedirs('my_car_data_28/train/right', exist_ok=True)
os.makedirs('my_car_data_28/train/straight', exist_ok=True)


def generate_dummy_scene():
    """生成28x28的极简道路场景"""
    img = np.zeros((28, 28), dtype=np.uint8)

    # 画两条车道线（位置按比例缩放）
    cv2.line(img, (7, 0), (7, 27), 200, 1)  # 左车道线
    cv2.line(img, (21, 0), (21, 27), 200, 1)  # 右车道线

    # 随机决定障碍物在哪条车道
    lane = np.random.choice(['left', 'right', 'none'], p=[0.33, 0.33, 0.34])

    if lane == 'left':
        # 障碍物放在左车道中间
        cv2.circle(img, (10, 14), 3, 255, -1)
        label = 'right'  # 障碍在左 → 右转
    elif lane == 'right':
        # 障碍物放在右车道中间
        cv2.circle(img, (18, 14), 3, 255, -1)
        label = 'left'  # 障碍在右 → 左转
    else:
        label = 'straight'

    return img, label


# 生成1000张训练图
print("开始生成28x28数据...")
for i in range(1000):
    img, label = generate_dummy_scene()
    cv2.imwrite(f'my_car_data_28/train/{label}/img_{i:04d}.png', img)
    if i % 100 == 0:
        print(f'已生成 {i} 张图')
print("数据生成完成！")