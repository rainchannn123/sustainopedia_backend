# LCA Conda环境包（含完整数据库）

## 概述

这是一个包含完整数据库的LCA（生命周期评估）计算环境，支持两个独立的数据库系统：
- **Tiangong数据库** (Python 3.11 + Brightway2 3.6.6)
- **Ecoinvent数据库** (Python 3.10 + Brightway2 2.4.0+)

 
## 快速开始

### 1. 环境要求
- **Anaconda** 或 **Miniconda** (推荐最新版本)
- **内存**: 至少8GB RAM
- **磁盘空间**: 至少15GB可用空间（包含数据库）

### 2. 安装步骤

#### 步骤1: 解压环境包
```bash
tar -xzf lca-conda-complete.tar.gz
cd lca-conda-complete
```

#### 步骤2: 运行安装脚本
```bash
./install.sh
```

#### 步骤3: 重新加载环境
```bash
source ~/.bashrc
```

## 使用方法

### 方法1: 使用便捷别名（推荐）

#### 激活环境
```bash
activate-tiangong    # 激活Tiangong环境
activate-ecoinvent   # 激活Ecoinvent环境
```

#### 单文件计算
```bash
# Tiangong计算
calc-tiangong test_tiangong.json

# Ecoinvent计算
calc-ecoinvent test.json
```

#### 批量计算
```bash
# Tiangong批量计算
batch-tiangong test_files/

# Ecoinvent批量计算
batch-ecoinvent test_files/
```

### 方法2: 使用切换脚本

```bash
# 切换到Tiangong环境
./switch_to_tiangong.sh

# 切换到Ecoinvent环境
./switch_to_ecoinvent.sh
```

### 方法3: 直接使用Python

#### Tiangong计算
```bash
# 激活环境
activate-tiangong

# 运行计算
python lca_calculator.py tiangong test_tiangong.json
```

#### Ecoinvent计算
```bash
# 激活环境
activate-ecoinvent

# 运行计算
python lca_calculator.py ecoinvent test.json
```

## 输入文件格式

### JSON输入格式示例
```json
{
  "goal": "lifecycle impact assessment",
  "method": ["Tiangong", "Environmental_Footprint", "Acidification"],
  "system_boundary": "cradle-to-grave",
  "functional_unit": [
    {
      "activity_name": "iron ore mining",
      "amount": 1.0,
      "unit": "ton",
      "location": "CN",
      "stage": "raw material extraction"
    }
  ]
}
```

## 测试和验证

### 1. 环境验证
```bash
# 验证Tiangong环境
activate-tiangong
python -c "import bw2data; print('Brightway2版本:', bw2data.__version__)"

# 验证Ecoinvent环境
activate-ecoinvent
python -c "import bw2data; print('Brightway2版本:', bw2data.__version__)"
```

### 2. 数据库验证
```bash
# 检查Tiangong数据库
activate-tiangong
python -c "from bw2data import databases; print('Tiangong数据库:', 'Tiangong' in databases)"

# 检查Ecoinvent数据库
activate-ecoinvent
python -c "from bw2data import databases; print('Ecoinvent数据库:', 'ecoinvent' in databases)"
```

## 故障排除

### 常见问题及解决方案

#### 1. Conda环境激活失败
```bash
conda init bash
source ~/.bashrc
```

#### 2. 数据库初始化失败
```bash
# 重新初始化Tiangong数据库
activate-tiangong
python init_databases.py tiangong

# 重新初始化Ecoinvent数据库
activate-ecoinvent
python init_databases.py ecoinvent
```

#### 3. 内存不足
```bash
# 检查内存使用
free -h  # Linux
vm_stat  # macOS
```

## 文件结构

```
lca-conda-complete/
├── install.sh                    # 主安装脚本
├── setup_environments.sh         # 环境设置脚本
├── switch_to_tiangong.sh         # Tiangong环境切换
├── switch_to_ecoinvent.sh        # Ecoinvent环境切换
├── init_databases.py             # 数据库初始化脚本
├── lca_calculator.py             # 单文件计算器
├── batch_calculator.py           # 批量计算器
├── Unified_LCA_Engine.py         # 统一LCA引擎
├── Tiangong_Backend.py          # Tiangong后端
├── Ecoinvent_Backend.py         # Ecoinvent后端
├── init_tiangong_db.py          # Tiangong数据库初始化
├── init_database.py             # Ecoinvent数据库初始化
├── test_tiangong.json           # Tiangong测试文件
├── test.json                    # Ecoinvent测试文件
├── test_files/                  # 更多测试文件
├── tiangong-lca-data-v0.2.0/   # Tiangong原始数据
├── ecoinvent_data/              # Ecoinvent原始数据
├── brightway2-data/            # Brightway2数据库
└── README.md                    # 本说明文档
```

 