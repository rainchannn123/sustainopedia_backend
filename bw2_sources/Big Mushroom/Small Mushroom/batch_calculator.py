#!/usr/bin/env python3
"""
批量LCA计算器
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path

def batch_calculate(json_files, engine_type):
    """批量计算多个JSON文件"""
    print(f"=== 批量计算 ({engine_type}) ===")
    
    # 设置环境
    os.environ['ENGINE_TYPE'] = engine_type
    if engine_type == 'tiangong':
        os.environ['BRIGHTWAY2_DIR'] = "./brightway_databases/brightway2_tiangong"
    else:
        os.environ['BRIGHTWAY2_DIR'] = "./brightway_databases/brightway2_ecoinvent"
    
    try:
        from Unified_LCA_Engine import UnifiedLCAEngine
        
        engine = UnifiedLCAEngine()
        results = []
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n处理文件 {i}/{len(json_files)}: {json_file}")
            
            try:
                result = engine.run_lca_calculation(json_file)
                results.append({
                    'file': json_file,
                    'status': 'success',
                    'total_impact': result.get('total_impact', 'N/A'),
                    'method': result.get('method', 'N/A'),
                    'result': result
                })
                print(f"✓ 成功: {json_file}")
                
            except Exception as e:
                results.append({
                    'file': json_file,
                    'status': 'error',
                    'error': str(e),
                    'result': None
                })
                print(f"✗ 失败: {json_file} - {e}")
        
        return results
        
    except Exception as e:
        print(f"✗ 批量计算失败: {e}")
        return []

def export_results(results, output_file):
    """导出结果到Excel"""
    df_data = []
    
    for result in results:
        if result['status'] == 'success':
            df_data.append({
                'File': result['file'],
                'Status': result['status'],
                'Total_Impact': result['total_impact'],
                'Method': result['method']
            })
        else:
            df_data.append({
                'File': result['file'],
                'Status': result['status'],
                'Error': result['error']
            })
    
    df = pd.DataFrame(df_data)
    df.to_excel(output_file, index=False)
    print(f"结果已导出到: {output_file}")

def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python batch_calculator.py <engine_type> <json_directory>")
        print("engine_type: tiangong 或 ecoinvent")
        print("json_directory: 包含JSON文件的目录")
        print("")
        print("示例:")
        print("  python batch_calculator.py tiangong test_files/")
        print("  python batch_calculator.py ecoinvent test_files/")
        return
    
    engine_type = sys.argv[1]
    json_dir = sys.argv[2]
    
    # 检查目录是否存在
    if not os.path.exists(json_dir):
        print(f"错误: 目录 {json_dir} 不存在")
        return
    
    # 查找JSON文件
    json_files = list(Path(json_dir).glob("*.json"))
    if not json_files:
        print(f"错误: 目录 {json_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 运行批量计算
    results = batch_calculate(json_files, engine_type)
    
    # 导出结果
    output_file = f"batch_results_{engine_type}.xlsx"
    export_results(results, output_file)
    
    # 打印摘要
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    
    print(f"\n=== 批量计算完成 ===")
    print(f"总文件数: {len(results)}")
    print(f"成功: {success_count}")
    print(f"失败: {error_count}")
    print(f"结果文件: {output_file}")

if __name__ == "__main__":
    main()
