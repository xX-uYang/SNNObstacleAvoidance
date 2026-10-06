import os
import re
from pathlib import Path

TARGET_DIR = "scripts"
CORE_MODULES = [
    'config', 'config_real', 'config_small',
    'dataset_aug', 'dataset_color', 'dataset_general',
    'conv_snn_28', 'flexible_conv_snn', 'advanced_snn'
]

def fix_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 插入路径修复代码（确保能找到 core 文件夹）
    header = """import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""
    if "PROJECT_ROOT = Path(__file__).parent.parent" not in content:
        content = header + content
    
    # 替换导入语句：import config -> from core import config
    for mod in CORE_MODULES:
        content = re.sub(r'^import\s+' + mod + r'\b', f'from core import {mod}', content, flags=re.MULTILINE)
        content = re.sub(r'^from\s+' + mod + r'\b\s+import', f'from core.{mod} import', content, flags=re.MULTILINE)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 已修复: {file_path.name}")

if __name__ == "__main__":
    for f in Path(TARGET_DIR).glob("*.py"):
        fix_file(f)
    print("🎉 全部修复完成！")