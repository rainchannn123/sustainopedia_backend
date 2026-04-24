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
import zipfile
import posixpath


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # 可选: DEBUG, INFO, WARNING, ERROR, CRITICAL
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ecoinvent_parse.log')
    ]
)

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:
    logging.info("请先安装 Azure SDK: pip install azure-storage-blob")
    sys.exit(1)
    
    
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
    """初始化Ecoinvent数据库"""
    logging.info("=== 初始化Ecoinvent数据库 ===")
    
    # 设置 Brightway2 环境（本地目录不变，Brightway2 仍需要本地路径）
    os.environ['ENGINE_TYPE'] = 'ecoinvent'
    os.environ['BRIGHTWAY2_DIR'] = "./brightway_databases/brightway2_ecoinvent"
    
    bw2_dir = os.environ['BRIGHTWAY2_DIR']
    os.makedirs(bw2_dir, exist_ok=True)
        
    # ====================== 新增：从 Azure Blob 下载 ======================
    # 请把以下变量改成你的实际值（推荐通过环境变量传入，避免硬编码）
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        logging.error("AZURE_STORAGE_CONNECTION_STRING env var is not set — cannot download database.")
        return False

    container_name = os.environ.get("AZURE_CONTAINER_NAME")
    zip_blob_name  = os.environ.get("ECOINVENT_ZIP_BLOB")

    MARKER_FILE = os.path.join(bw2_dir, ".ecoinvent_extracted_complete")
    
    if os.path.exists(MARKER_FILE):
        logging.info(f"本地数据库已存在且完整 ({bw2_dir})，跳过 Azure 下载")
        return True

    # 如果目录存在但没有标记文件 → 可能是之前 partial，先清空
    if os.path.exists(bw2_dir):
        logging.info("检测到未完成的历史解压，清理目录后重新下载...")
        for root, dirs, files in os.walk(bw2_dir, topdown=False):
            for f in files:
                os.remove(os.path.join(root, f))
            for d in dirs:
                os.rmdir(os.path.join(root, d))
    else:
        os.makedirs(bw2_dir, exist_ok=True)

    logging.info(f"从 Azure Blob Storage 下载 Ecoinvent 数据库...")
    local_zip = os.path.join(bw2_dir, "temp_ecoinvent.zip")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=zip_blob_name
        )

        logging.info("开始下载 zip 文件（分块流式下载，更省内存）...")
        download_stream = blob_client.download_blob()
        with open(local_zip, "wb") as f:
            for chunk in download_stream.chunks():   # ← 关键改动：流式下载
                f.write(chunk)

        logging.info("下载完成，开始解压...")
        
        with zipfile.ZipFile(local_zip, 'r') as z:
        # Step 1: 计算共同前缀（使用 posixpath，保证 Windows/Linux 一致）
            all_filenames = [info.filename for info in z.infolist() if not info.is_dir()]
            if not all_filenames:
                raise ValueError("Zip 文件中没有找到任何文件")

            common_prefix = posixpath.commonpath(all_filenames)
            if common_prefix and not common_prefix.endswith('/'):
                common_prefix += '/'

            logging.info(f"检测到 zip 内共同前缀: '{common_prefix}'（如果为空则无嵌套）")

            # Step 2: 解压 + 自动去掉共同前缀
            extracted_count = 0
            for info in z.infolist():
                if info.is_dir():
                    continue

                if not info.filename.startswith(common_prefix):
                    logging.warning(f"跳过不匹配路径: {info.filename}")
                    continue

                relative_path = info.filename[len(common_prefix):]
                target_path = os.path.join(bw2_dir, relative_path)

                os.makedirs(os.path.dirname(target_path), exist_ok=True)

                with z.open(info) as src, open(target_path, 'wb') as dst:
                    dst.write(src.read())

                extracted_count += 1

            if extracted_count == 0:
                raise ValueError("没有成功解压任何文件 — 检查 zip 结构")

            logging.info(f"成功解压 {extracted_count} 个文件 → 总大小应为完整 1.73 GB")

        # 解压成功 → 创建完成标记，下次不再重复下载
        with open(MARKER_FILE, "w") as f:
            f.write("extracted")

            logging.info(f"成功解压 {extracted_count} 个文件")

        os.remove(local_zip)
        logging.info(f"解压完成!Ecoinvent 数据库已就绪：{bw2_dir}")
        
    except Exception as e:
        logging.error(f"下载/解压失败: {e}")
        # 清理可能产生的残留文件，避免下次又被当成“已存在”
        if os.path.exists(local_zip):
            os.remove(local_zip)
        raise
    
    # try:
    #     # 复制预构建的数据库
    #     source_dir = "brightway2-data"
    #     if os.path.exists(source_dir):
    #         print(f"复制Ecoinvent数据库从 {source_dir} 到 {bw2_dir}")
    #         for item in os.listdir(source_dir):
    #             src = os.path.join(source_dir, item)
    #             dst = os.path.join(bw2_dir, item)
    #             if os.path.isdir(src):
    #                 shutil.copytree(src, dst, dirs_exist_ok=True)
    #             else:
    #                 shutil.copy2(src, dst)
        
    #     # 运行初始化脚本
        
    #     print("✓ Ecoinvent数据库初始化完成")
    #     return True
    # except Exception as e:
    #     print(f"✗ Ecoinvent数据库初始化失败: {e}")
    #     return False

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
