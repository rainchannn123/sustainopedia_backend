import os
import glob
import logging
import sys
import warnings
from bw2data import projects, databases, Database, Method
from bw2io import create_default_biosphere3
from bw2io.importers import SingleOutputEcospold2Importer, EcoinventLCIAImporter
import functools
import json
import itertools

# 忽略特定警告
warnings.filterwarnings('ignore', message='No valid output stream')

# 禁用 tqdm 进度条
os.environ["TQDM_DISABLE"] = "1"
# 强制使用非交互式模式
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
import time
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_databases(method=None):
    """初始化数据库"""
    logger.info("检查环境设置...")
    logger.info(f"Python版本: {os.sys.version}")
    logger.info(f"当前工作目录: {'/app'}")
    logger.info(f"Brightway2数据目录: {os.getenv('BRIGHTWAY2_DIR')}")
    
    # 检查项目
    logger.info(f"可用项目: {projects}")

    # 清理旧项目
    for project in ['ecoinvent', 'Ecoinvent']:
        if project in projects:
            logger.info(f"删除旧项目: {project}")
            projects.delete_project(project)
    
    # 创建新项目
    projects.create_project("ecoinvent")
    projects.set_current("ecoinvent")
    logger.info("创建并设置新项目: ecoinvent")
    logger.info(f"Project directory after set: {projects.dir}")
    
    # 重新加载数据库
    # databases.clean()
    # logger.info("清理并重新加载数据库...")
    
    # 确保 biosphere3 存在
    #########HERE WE GO #########
    # del databases["biosphere3"]
    # del databases["ecoinvent"]        
    if "biosphere3" not in databases:
        create_default_biosphere3()
        logger.info("创建 biosphere3 数据库")
    else:
        logger.info("biosphere3 已存在")
    
    logger.info(f"Project directory: {projects.dir}")
    logger.info(f"Current project: {projects.current}")

    logger.info(f"可用数据库: {list(databases)}")
    
    # 检查是否需要导入
    if "ecoinvent 3.8 cutoff" in databases:
        db = Database("ecoinvent 3.8 cutoff")
        activity_count = sum(1 for _ in db)
        logger.info(f"数据库已存在，包含 {activity_count} 个活动")
        
        # 如果数据库为空，需要重新导入
        if activity_count == 0:
            logger.info("数据库为空，需要重新导入...")
            del databases["ecoinvent 3.8 cutoff"]
        else:
            # 检查一些示例活动（只枚举前几个，避免加载整个数据库）
            for i, act in enumerate(itertools.islice(db, 5)):
                logger.info(f"示例活动 {i+1}: {act.get('name', 'Unknown')} ({act.get('location', 'Unknown')})")
            return True
        
    # 如果数据库不存在，则需要导入
    datasets_path = './bw2_sources/Big Mushroom/Small Mushroom/ecoinvent_data'
    db_name = "ecoinvent"
    
    # 检查文件权限
    spold_files = glob.glob(os.path.join(datasets_path, "*.spold"))
    logger.info(f"找到 {len(spold_files)} 个 .spold 文件")
    
    logger.info("检查文件权限...")
    for file in spold_files[:5]:
        if os.access(file, os.R_OK):
            logger.info(f"文件可读: {file}")
        else:
            logger.error(f"文件不可读: {file}")
    
    # Manifest optimization: skip work if nothing changed since last import
    # try:
    #     manifest = os.path.join(datasets_path, ".import_manifest.json")
    #     latest_mtime = max((os.path.getmtime(p) for p in spold_files), default=0)
    #     spold_count = len(spold_files)
    #     if os.path.exists(manifest):
    #         with open(manifest, "r", encoding="utf-8") as fh:
    #             mdata = json.load(fh)
    #         if mdata.get("spold_count") == spold_count and mdata.get("last_mtime") == latest_mtime and db_name in databases:
    #             db = Database(db_name)
    #             activity_count = sum(1 for _ in db)
    #             expected_activities = spold_count
    #             if activity_count >= expected_activities * 0.9:
    #                 logger.info("未检测到数据变化，跳过导入 (manifest matched)")
    #                 return True
    # except Exception as e:
    #     logger.warning(f"Manifest check failed, will continue with import: {e}")
    
    try:
        logger.info(f"从 {datasets_path} 导入 {db_name}...")
        #?
        DEFAULT_METHOD = ('IPCC 2021 no LT',
          'climate change: fossil, including SLCFs no LT',
          'global warming potential (GWP100) no LT')
        
        if method is None:
            method = DEFAULT_METHOD
            

        m = Method(method)
        logging.info(f"method::::{m}")
        loaded_cf = list(m.load())
        logging.info(f"CF count: {len(loaded_cf)}")
        # cf_dict = {cf[0]: cf[1] for cf in loaded_cf}
        # logging.info(f"Sample CFs: {list(itertools.islice(cf_dict.items(), 10))}")
        # 检查数据库是否已存在
        if db_name in databases:
            db = Database(db_name)
            activity_count = sum(1 for _ in db)
            expected_activities = len(spold_files)
            bios = []
            logger.info(f"Foreground biosphere count: {len(bios)}")

            if activity_count >= expected_activities * 0.9:  # 如果已导入90%以上的数据
                logger.info(f"数据库 {db_name} 已存在且包含 {activity_count} 个活动（预期 {expected_activities} 个），跳过导入")
                return True
            else:
                del databases[db_name]
                logger.info(f"删除不完整的数据库: {db_name}，当前活动数 {activity_count}，预期活动数 {expected_activities}")
        
        # 创建临时目录
        temp_dir = './bw2_sources/Big Mushroom/Small Mushroom/temp/spold_files'
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
        
        # 直接使用内置的数据目录
        logger.info("开始处理数据...")
        datasets_dir = './bw2_sources/Big Mushroom/Small Mushroom/ecoinvent_data'
        logger.info(f"数据目录: {datasets_dir}")
        logger.info(f"文件数量: {len(spold_files)}")
        
        # 检查一些示例文件
        for file in spold_files[:5]:
            logger.info(f"示例文件: {os.path.basename(file)}")
        
        # # NEW 检查数据集变更清单
        # manifest = os.path.join(datasets_dir, ".import_manifest.json")
        # latest_mtime = max(os.path.getmtime(p) for p in spold_files)
        # if os.path.exists(manifest):
        #     m = json.load(open(manifest))
        #     if m.get("spold_count") == len(spold_files) and m.get("last_mtime") == latest_mtime and db_name in databases:
        #         logger.info("未检测到数据集更改 - 跳过导入")
        #         return True
        
        try:
            # 创建导入器
            logger.info("创建导入器...")
            sys.stdout.write("开始导入数据...\n")
            sys.stdout.flush()
            
            # 创建导入器并配置 (启用多进程如果系统支持，并在失败时回退到单进程)
            use_mp = os.environ.get("IMPORT_USE_MP", "1") == "1" and (os.cpu_count() or 1) > 1
            try:
                importer = SingleOutputEcospold2Importer(datasets_dir, db_name, use_mp=use_mp)
                logger.info(f"创建导入器 (use_mp={use_mp})...")
            except Exception as e:
                logger.warning(f"创建使用多进程导入器失败，回退到单进程：{e}")
                importer = SingleOutputEcospold2Importer(datasets_dir, db_name, use_mp=False)
            # 设置导入器的输出流
            importer.output_stream = sys.stdout
            
            # 应用导入策略
            logger.info("应用导入策略...")
            importer.apply_strategies()
            unlinked = [exc for ds in importer.data for exc in ds.get('exchanges', []) if exc['type'] == 'biosphere' and 'input' not in exc]
            logger.info(f"Unlinked biosphere exchanges: {len(unlinked)}")
            for exc in unlinked[:5]:  # Sample
                logger.info(f"Unlinked: name={exc.get('name')}, categories={exc.get('categories')}, unit={exc.get('unit')}")
            logger.info(f"{importer.statistics()}")
            logger.info(f"导入器数据集数量: {len(importer.data)}")

            # 检查数据集
            for i, ds in enumerate(importer.data[:5]):
                logger.info(f"示例数据集 {i+1}: {ds.get('name', 'Unknown')} ({ds.get('location', 'Unknown')})")
            
            
            biosphere = Database("biosphere3")
            logger.info(f"Number of flows: {len(biosphere)}")

            # 写入数据库
            logger.info("写入数据库...")
            importer.match_database("biosphere3", fields=['name', 'unit', 'categories'])
            logger.info(f"Still unlinked after strict match: {len(list(importer.unlinked))}")
            
            if importer.unlinked:
                importer.match_database("biosphere3", fields=['name','categories'])
                logger.info(f"Still unlinked after name+category: {len(list(importer.unlinked))}")
                
            if importer.unlinked:
                importer.match_database("biosphere3", fields=['name', 'unit'])
                logger.info(f"Still unlinked after name+unit: {len(list(importer.unlinked))}")
            # Only add if truly missing
            if importer.unlinked:
                logger.info("Adding unlinked flows to biosphere3...")
                removed_count = 0
                for ds in importer.data:
                    original_ex = len(ds['exchanges'])
                    ds['exchanges'] = [exc for exc in ds['exchanges'] if 'input' in exc]
                    removed_count += original_ex - len(ds['exchanges'])
                logger.info(f"Removed {removed_count} unlinked exchanges")

            logger.info(f"After final unlinked flow matching: {importer.statistics()}")
            importer.write_database()
            logger.info("数据库写入完成")
            
            # 写入导入 manifest，用于快速跳过不必要的后续导入
            # try:
            #     manifest = os.path.join(datasets_dir, ".import_manifest.json")
            #     latest_mtime = max((os.path.getmtime(p) for p in spold_files), default=0)
            #     with open(manifest, "w", encoding="utf-8") as fh:
            #         json.dump({"spold_count": len(spold_files), "last_mtime": latest_mtime, "db": db_name}, fh)
            #     logger.info(f"Wrote import manifest: {manifest}")
            # except Exception as e:
            #     logger.warning(f"写入 manifest 失败: {e}")
            
            # 验证数据库
            db = Database(db_name)
            activity_sample = list(itertools.islice(db, 5))
            activity_count = sum(1 for _ in db)
            logger.info(f"数据库导入完成，包含 {activity_count} 个活动")
            
            # 检查一些示例活动（只列举前几个）
            for i, act in enumerate(activity_sample):
                logger.info(f"示例活动 {i+1}: {act.get('name', 'Unknown')} ({act.get('location', 'Unknown')})")
                
            # 数据库初始化完成
            logger.info("数据库初始化完成，准备进行测试...")
            
        except Exception as e:
            logger.error(f"导入过程中出错: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
        
        # 重新加载数据库
        # databases.clean()
        # db = Database(db_name)
        # db_count = sum(1 for _ in db)
        # logger.info(f"数据库 {db_name} 导入成功，活动数量: {db_count}")
        
        # 打印所有活动
        # for act in db:
        #     logger.info(f"Activity in DB: {act['name']} ({act['location']})")
        
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        # 更新数据集变更清单
        # json.dump({"spold_count": len(spold_files), "last_mtime": latest_mtime, "db": db_name}, open(manifest, "w"))
        
        return True
        
    except Exception as e:
        logger.error(f"导入过程中出错: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    init_databases()