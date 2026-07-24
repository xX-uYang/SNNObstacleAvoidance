import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 配置
data_root = "D:/SNN/260417_real_data/train"  # 原数据集根目录（包含 left, right, straight）
output_root = "D:/SNN/260427_real_data_split"  # 划分后的输出根目录
train_ratio = 0.8  # 训练集比例
random.seed(42)  # 固定随机种子，确保可复现

# 创建输出目录
train_root = os.path.join(output_root, 'train')
test_root = os.path.join(output_root, 'test')
os.makedirs(train_root, exist_ok=True)
os.makedirs(test_root, exist_ok=True)

# 遍历每个类别
for cls in ['left', 'right', 'straight']:
    src_dir = os.path.join(data_root, cls)
    if not os.path.exists(src_dir):
        print(f"警告：{src_dir} 不存在，跳过")
        continue

    # 获取该类下所有文件
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(files) == 0:
        print(f"警告：{src_dir} 中没有图片文件")
        continue

    # 划分训练集和测试集
    train_files, test_files = train_test_split(files, train_size=train_ratio, random_state=42)

    # 创建类别子目录（train 和 test 下都要有）
    train_cls_dir = os.path.join(train_root, cls)
    test_cls_dir = os.path.join(test_root, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)

    # 复制训练集文件
    for f in train_files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(train_cls_dir, f)
        shutil.copy(src, dst)

    # 复制测试集文件
    for f in test_files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(test_cls_dir, f)
        shutil.copy(src, dst)

    print(f"{cls}: 训练集 {len(train_files)} 张，测试集 {len(test_files)} 张")

print("数据集划分完成！")
print(f"训练集路径: {train_root}")
print(f"测试集路径: {test_root}")