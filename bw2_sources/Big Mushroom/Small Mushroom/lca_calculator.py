#!/usr/bin/env python3
"""
LCA计算器 - 直接使用Python进行LCA计算
"""
import os
import sys
import json
from pathlib import Path

def run_tiangong_calculation(json_file):
    """运行Tiangong LCA计算"""
    print("=== Tiangong LCA计算 ===")
    
    # 设置环境
    os.environ['ENGINE_TYPE'] = 'tiangong'
    os.environ['BRIGHTWAY2_DIR'] = "./brightway_databases/brightway2_tiangong"
    
    try:
        from Unified_LCA_Engine import UnifiedLCAEngine
        
        # 创建引擎
        engine = UnifiedLCAEngine()
        
        # 运行计算
        result = engine.run_lca_calculation(json_file)
        
        print("✓ Tiangong计算完成")
        return result
        
    except Exception as e:
        print(f"✗ Tiangong计算失败: {e}")
        return None

def run_ecoinvent_calculation(json_file):
    """运行Ecoinvent LCA计算"""
    print("=== Ecoinvent LCA计算 ===")
    
    # 设置环境
    os.environ['ENGINE_TYPE'] = 'ecoinvent'
    os.environ.setdefault('BRIGHTWAY2_DIR', "./brightway_databases/brightway2_ecoinvent")
    
    try:
        from Unified_LCA_Engine import UnifiedLCAEngine
        
        # 创建引擎
        engine = UnifiedLCAEngine()
        
        # 运行计算
        result = engine.run_lca_calculation(json_file)
        
        print("✓ Ecoinvent计算完成")
        return result
        
    except Exception as e:
        print(f"✗ Ecoinvent计算失败: {e}")
        return None

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python lca_calculator.py <engine_type> <json_file>")
        print("engine_type: tiangong 或 ecoinvent")
        print("json_file: JSON输入文件路径")
        print("")
        print("示例:")
        print("  python lca_calculator.py tiangong test_tiangong.json")
        print("  python lca_calculator.py ecoinvent test.json")
        return
    
    engine_type = sys.argv[1]
    json_file = sys.argv[2]
    
    # 检查文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 文件 {json_file} 不存在")
        return
    
    # 读取JSON文件
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"读取文件: {json_file}")
    except Exception as e:
        print(f"错误: 无法读取文件 {json_file}: {e}")
        return
    
    # 运行计算
    if engine_type == 'tiangong':
        run_tiangong_calculation(json_file)
    elif engine_type == 'ecoinvent':
        run_ecoinvent_calculation(json_file)
    else:
        print(f"错误: 不支持的引擎类型 {engine_type}")
        print("支持的引擎类型: tiangong, ecoinvent")

if __name__ == "__main__":
    main()
