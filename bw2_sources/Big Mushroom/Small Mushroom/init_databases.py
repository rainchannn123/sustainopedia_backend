#!/usr/bin/env python3
"""
初始化两个环境的数据库
"""
import os
import sys
import shutil
from pathlib import Path
from init_tiangong_db import *
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ecoinvent_parse.log')
    ]
)

def init_tiangong():
    """初始化Tiangong数据库"""
    print("=== 初始化Tiangong数据库 ===")
    
    # 设置环境
    os.environ['ENGINE_TYPE'] = 'tiangong'
    os.environ['BRIGHTWAY2_DIR'] = "./brightway_databases/brightway2_tiangong"
    
    # 确保数据目录存在
    bw2_dir = os.environ['BRIGHTWAY2_DIR']
    os.makedirs(bw2_dir, exist_ok=True)
    
    
    try:
        # 复制预构建的数据库
        source_dir = "brightway2-data"
        if os.path.exists(source_dir):
            print(f"复制Tiangong数据库从 {source_dir} 到 {bw2_dir}")
            for item in os.listdir(source_dir):
                src = os.path.join(source_dir, item)
                dst = os.path.join(bw2_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        
        # 运行初始化脚本
        
        print("✓ Tiangong数据库初始化完成")
        return True
    except Exception as e:
        print(f"✗ Tiangong数据库初始化失败: {e}")
        return False

def init_ecoinvent():
    """Restore the prebuilt Ecoinvent database from MongoDB."""
    backend_root = Path(__file__).resolve().parents[3]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    from Generation.functions.prebuild_bw2db import preinstall_ecoinvent_database

    os.environ["ENGINE_TYPE"] = "ecoinvent"
    return preinstall_ecoinvent_database()

if __name__ == "__main__":
    print("初始化LCA数据库...")
    
    # 检查参数
    if len(sys.argv) > 1:
        env_type = sys.argv[1]
        if env_type == 'tiangong':
            init_tiangong()
        elif env_type == 'ecoinvent':
            init_ecoinvent()
        else:
            print("用法: python init_databases.py [tiangong|ecoinvent]")
    else:
        # 初始化所有环境
        init_tiangong()
        init_ecoinvent()
