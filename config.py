"""
配置文件 - 黄海海水温度预测模型
"""
import os
from datetime import datetime

# Google Earth Engine配置 (保留占位符，当前数据收集改为开放API)
GEE_PROJECT = 'your-gee-project-id'  # 兼容旧流程，当前未使用
GEE_SERVICE_ACCOUNT = 'your-service-account@your-project.iam.gserviceaccount.com'

# 黄海区域坐标 (经纬度边界)
YELLOW_SEA_BOUNDS = {
    'west': 119.0,   # 西经
    'east': 127.0,   # 东经  
    'south': 32.0,   # 南纬
    'north': 41.0    # 北纬
}

# 数据收集配置
DATA_CONFIG = {
    'start_date': '2018-01-01',
    'end_date': datetime.now().strftime('%Y-%m-%d')
}

# 开放海洋温度API（Open-Meteo Marine Archive）配置
OPEN_METEO_CONFIG = {
    'base_url': 'https://archive-api.open-meteo.com/v1/marine',
    'daily_variables': [
        'sea_surface_temperature_mean',
        'sea_surface_temperature_min',
        'sea_surface_temperature_max'
    ],
    'grid_lat_points': 5,
    'grid_lon_points': 5,
    'request_chunk_days': 31  # 每次请求的最大天数，防止URL过长或API限速
}

# PINN模型配置
MODEL_CONFIG = {
    'hidden_layers': [128, 128, 64, 32],
    'activation': 'tanh',
    'learning_rate': 0.001,
    'epochs': 1000,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping_patience': 50
}

# 物理约束配置
PHYSICS_CONFIG = {
    'temperature_range': (-3, 31),   # 黄海实际 SST 物理范围（°C），冬季可低于 0
    'seasonal_cycle': True,
    'spatial_smoothness': True,
    'conservation_law': True,

    # 各约束损失权重
    'lambda_range': 1.0,
    'lambda_seasonal': 0.1,
    'lambda_smooth': 0.01,
    'lambda_conservation': 0.05,

    # 物理损失 warm-up：前 N 轮线性增加权重，避免过早约束
    'warmup_epochs': 30,

    # 热惯性：海表温度日变化上限（°C/天）
    'max_daily_change_celsius': 2.0,
}

# 输出配置
OUTPUT_CONFIG = {
    'model_save_path': 'models/',
    'data_save_path': 'data/',
    'results_save_path': 'results/',
    'plots_save_path': 'plots/'
}

# NetCDF配置
NETCDF_CONFIG = {
    'enabled': True,  # 是否启用NetCDF支持
    'default_format': 'both',  # 默认保存格式: 'csv', 'netcdf', 'both'
    'spatial_grid': {
        'lat_points': 5,  # 纬度网格点数
        'lon_points': 5,  # 经度网格点数
        'create_grid': True  # 是否创建空间网格
    },
    'compression': {
        'enabled': True,  # 是否启用压缩
        'level': 4  # 压缩级别 (0-9)
    }
}

# 预处理/特征开关配置
PREPROCESSING_CONFIG = {
    'enable_lag_features': True,
    'lag_features': [1, 2, 3, 7, 14, 30],
    'enable_moving_statistics': False,
    'moving_windows': [3, 7, 14, 30],
    'enable_derived_features': False,
    'enable_physics_features': False,
    'enable_interaction_features': False
}

# GCOM-C / SGLI L3 SST V3：由 sgli_l3_sst_extract_gee.py 生成的训练用 CSV（推荐）
SGLI_DATA_CONFIG = {
    'csv_relative_path': 'data/sgli_yellow_sea_sst_daily.csv',
    # Earth Engine 初始化用；也可通过环境变量 EE_PROJECT 设置
    'ee_project': os.environ.get('EE_PROJECT', ''),
}


def resolve_training_data_csv() -> str:
    """
    训练默认数据路径：优先使用 GCOM-C/SGLI L3 日值 CSV，否则使用原有演示/采集路径。
    """
    sgli_path = SGLI_DATA_CONFIG['csv_relative_path']
    legacy_path = os.path.join(OUTPUT_CONFIG['data_save_path'], 'yellow_sea_sst_data.csv')
    if os.path.exists(sgli_path):
        return sgli_path
    return legacy_path


# 创建必要的目录
for path in OUTPUT_CONFIG.values():
    os.makedirs(path, exist_ok=True)



