import os
import sys
import shutil
from pathlib import Path
import zipfile
import posixpath
from azure.storage.blob import BlobServiceClient

loadenv = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(loadenv):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=loadenv)
        
bw2_dir = os.path.join(os.path.dirname(__file__), 'bw2_data')
os.makedirs(bw2_dir, exist_ok=True)
local_zip = os.path.join(os.path.dirname(__file__), 'temp_ecoinvent.zip')  
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
blob_client = blob_service_client.get_blob_client(
    container=os.getenv("AZURE_CONTAINER_NAME"),
    blob=os.getenv("ECOINVENT_ZIP_BLOB")
)

download_stream = blob_client.download_blob()
with open(local_zip, "wb") as f:
    for chunk in download_stream.chunks():   # ← 关键改动：流式下载
        f.write(chunk)


with zipfile.ZipFile(local_zip, 'r') as z:
# Step 1: 计算共同前缀（使用 posixpath，保证 Windows/Linux 一致）
    all_filenames = [info.filename for info in z.infolist() if not info.is_dir()]
    if not all_filenames:
        raise ValueError("Zip 文件中没有找到任何文件")

    common_prefix = posixpath.commonpath(all_filenames)
    if common_prefix and not common_prefix.endswith('/'):
        common_prefix += '/'

    # Step 2: 解压 + 自动去掉共同前缀
    extracted_count = 0
    for info in z.infolist():
        if info.is_dir():
            continue

        if not info.filename.startswith(common_prefix):
            continue

        relative_path = info.filename[len(common_prefix):]
        target_path = os.path.join(bw2_dir, relative_path)

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with z.open(info) as src, open(target_path, 'wb') as dst:
            dst.write(src.read())

        extracted_count += 1

    if extracted_count == 0:
        raise ValueError("没有成功解压任何文件 — 检查 zip 结构")
 