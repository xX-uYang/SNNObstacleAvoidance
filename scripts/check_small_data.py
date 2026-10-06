import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from PIL import Image

root = 'D:/SNN/test_small_data/train'   # 改成你的实际路径
for cls in ['left', 'right', 'straight']:
    folder = os.path.join(root, cls)
    if not os.path.exists(folder):
        print(f'警告：{folder} 不存在！')
        continue
    print(f'\n--- {cls} 文件夹 ---')
    files = os.listdir(folder)
    print(f'图片数量：{len(files)}')
    for fname in files[:3]:   # 只看前3张
        path = os.path.join(folder, fname)
        img = Image.open(path)
        print(f'{fname}: 尺寸 {img.size}, 模式 {img.mode}')