"""
数据预处理和特征工程模块
处理缺失值，转换时间特征为周期性特征，为PINN模型准备数据
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import List, Dict, Optional

from config import PREPROCESSING_CONFIG

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or PREPROCESSING_CONFIG
        self.scaler = StandardScaler()
        self.temperature_scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)
        
        # 配置开关
        self.enable_lag_features = self.config.get('enable_lag_features', True)
        self.enable_moving_statistics = self.config.get('enable_moving_statistics', True)
        self.moving_windows: List[int] = self.config.get('moving_windows', [3, 7, 14, 30])
        self.enable_derived_features = self.config.get('enable_derived_features', True)
        self.enable_physics_features = self.config.get('enable_physics_features', True)
        self.enable_interaction_features = self.config.get('enable_interaction_features', True)
        self.lag_features: List[int] = self.config.get('lag_features', [1, 2, 3, 7, 14, 30])
        
    def load_data(self, filepath):
        """
        加载原始数据（支持CSV和NetCDF格式）
        
        Args:
            filepath: 文件路径（自动检测格式）
        
        Returns:
            pandas DataFrame
        """
        import os
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")
        
        # 根据文件扩展名选择加载方式
        if filepath.endswith('.nc') or filepath.endswith('.netcdf'):
            # 加载NetCDF格式
            try:
                from netcdf_handler import NetCDFHandler
                handler = NetCDFHandler()
                
                # 创建临时CSV文件
                temp_csv = filepath.replace('.nc', '_temp.csv').replace('.netcdf', '_temp.csv')
                handler.netcdf_to_csv(filepath, temp_csv)
                
                # 读取CSV
                df = pd.read_csv(temp_csv)
                df['date'] = pd.to_datetime(df['date'])
                
                # 清理临时文件
                if os.path.exists(temp_csv):
                    os.remove(temp_csv)
                
                print(f"从NetCDF文件加载数据: {len(df)} 条记录")
                return df
            except ImportError:
                print("警告: NetCDF处理器未找到，尝试使用xarray直接读取...")
                try:
                    import xarray as xr
                    ds = xr.open_dataset(filepath)
                    # 转换为DataFrame（假设只有时间维度）
                    df = ds.to_dataframe().reset_index()
                    if 'time' in df.columns:
                        df['date'] = pd.to_datetime(df['time'])
                        df = df.drop('time', axis=1)
                    print(f"从NetCDF文件加载数据: {len(df)} 条记录")
                    return df
                except Exception as e:
                    raise ValueError(f"无法加载NetCDF文件: {e}")
        else:
            # 加载CSV格式（默认）
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            print(f"从CSV文件加载数据: {len(df)} 条记录")
            return df
    
    def handle_missing_values(self, df):
        """处理缺失值"""
        print("处理缺失值...")
        
        # 检查缺失值
        missing_info = df.isnull().sum()
        print(f"缺失值统计:\n{missing_info}")
        
        # 使用KNN插值填充缺失值（仅对存在的列生效）
        numeric_columns = ['mean_sst', 'min_sst', 'max_sst', 'std_sst', 'valid_pixels']
        
        for col in numeric_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col] = self.imputer.fit_transform(df[[col]]).flatten()
        
        print("缺失值处理完成")
        return df
    
    def create_temporal_features(self, df):
        """创建时间特征"""
        print("创建时间特征...")
        
        # 基础时间特征
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # 周期性特征 (关键!)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # 季节特征
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # 冬季
            3: 1, 4: 1, 5: 1,   # 春季
            6: 2, 7: 2, 8: 2,   # 夏季
            9: 3, 10: 3, 11: 3  # 秋季
        })
        
        # 季节的周期性编码
        df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
        df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
        
        print("时间特征创建完成")
        return df
    
    def create_lag_features(self, df, target_col='mean_sst'):
        """创建滞后特征"""
        if not self.enable_lag_features or target_col not in df.columns:
            print("跳过滞后特征创建")
            return df

        print("创建滞后特征...")

        for lag in self.lag_features:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        if self.enable_moving_statistics:
            for window in self.moving_windows:
                df[f'{target_col}_ma_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_std_{window}'] = df[target_col].rolling(window=window).std()

        print("滞后特征创建完成")
        return df
    
    def create_derived_features(self, df):
        """创建衍生特征"""
        if not self.enable_derived_features:
            print("跳过衍生特征创建")
            return df

        print("创建衍生特征...")

        # 温度变化率
        if 'mean_sst' in df.columns:
            df['sst_change'] = df['mean_sst'].diff()
            df['sst_change_rate'] = df['sst_change'] / df['mean_sst'].shift(1)

        # 温度范围
        if 'max_sst' in df.columns and 'min_sst' in df.columns:
            df['sst_range'] = df['max_sst'] - df['min_sst']

        # 温度稳定性 (变异系数)
        if 'std_sst' in df.columns and 'mean_sst' in df.columns:
            df['sst_cv'] = df['std_sst'] / df['mean_sst']

        # 数据质量指标
        if 'valid_pixels' in df.columns:
            df['data_quality'] = df['valid_pixels'] / df['valid_pixels'].max()

        print("衍生特征创建完成")
        return df
    
    def remove_outliers(self, df, target_col='mean_sst', method='iqr'):
        """移除异常值"""
        print("移除异常值...")
        
        if method == 'iqr':
            Q1 = df[target_col].quantile(0.25)
            Q3 = df[target_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[target_col] < lower_bound) | (df[target_col] > upper_bound)
            print(f"发现 {outliers.sum()} 个异常值")
            
            # 可以选择移除或替换异常值
            df = df[~outliers].reset_index(drop=True)
        
        print("异常值处理完成")
        return df
    
    def scale_features(self, df, feature_columns, target_column='mean_sst'):
        """特征缩放"""
        print("特征缩放...")
        
        # 标准化特征
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])
        
        # 目标变量归一化到[0,1]
        df[target_column] = self.temperature_scaler.fit_transform(df[[target_column]]).flatten()
        
        print("特征缩放完成")
        return df

    def apply_scalers(self, df, feature_columns, target_column='mean_sst'):
        """使用已拟合的缩放器转换新数据"""
        if feature_columns:
            df[feature_columns] = self.scaler.transform(df[feature_columns])
        if target_column in df.columns:
            df[target_column] = self.temperature_scaler.transform(df[[target_column]]).flatten()
        return df
    
    def prepare_training_data(self, df, target_col='mean_sst', test_size=0.2):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 移除包含NaN的行
        df = df.dropna().reset_index(drop=True)
        
        # 选择特征列
        feature_columns = [col for col in df.columns if col not in ['date', target_col]]
        
        # 分离特征和目标
        X = df[feature_columns].values
        y = df[target_col].values
        
        # 时间序列分割 (保持时间顺序)
        split_idx = int(len(df) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def visualize_data(self, df, save_path='plots/'):
        """数据可视化"""
        print("生成数据可视化...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 时间序列图
        axes[0, 0].plot(df['date'], df['mean_sst'], alpha=0.7)
        axes[0, 0].set_title('黄海海水温度时间序列')
        axes[0, 0].set_xlabel('日期')
        axes[0, 0].set_ylabel('平均SST (°C)')
        
        # 季节性模式
        if 'month' in df.columns:
            monthly_sst = df.groupby('month')['mean_sst'].mean()
            axes[0, 1].plot(monthly_sst.index, monthly_sst.values, marker='o')
            axes[0, 1].set_title('月度平均海水温度')
            axes[0, 1].set_xlabel('月份')
            axes[0, 1].set_ylabel('平均SST (°C)')
        else:
            axes[0, 1].text(0.5, 0.5, '无月份特征', ha='center', va='center')
            axes[0, 1].axis('off')
        
        # 温度分布
        axes[1, 0].hist(df['mean_sst'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('海水温度分布')
        axes[1, 0].set_xlabel('SST (°C)')
        axes[1, 0].set_ylabel('频次')
        
        # 温度变化率
        if 'sst_change' in df.columns:
            axes[1, 1].plot(df['date'], df['sst_change'], alpha=0.7)
            axes[1, 1].set_title('海水温度变化率')
            axes[1, 1].set_xlabel('日期')
            axes[1, 1].set_ylabel('温度变化 (°C)')
        else:
            axes[1, 1].text(0.5, 0.5, '无变化率特征', ha='center', va='center')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}data_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("数据可视化完成")

class FeatureEngineer:
    """特征工程器"""
    
    @staticmethod
    def create_physics_features(df):
        """创建物理相关特征"""
        print("创建物理特征...")
        
        # 太阳辐射相关 (简化的日长计算)
        df['day_length'] = 12 + 4 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        
        # 热容量相关 (假设与温度相关)
        df['thermal_capacity'] = 1 / (1 + np.exp(-(df['mean_sst'] - 15) / 5))
        
        # 海洋混合层深度 (简化模型)
        df['mixed_layer_depth'] = 50 + 20 * np.sin(2 * np.pi * df['day_of_year'] / 365)
        
        print("物理特征创建完成")
        return df
    
    @staticmethod
    def create_interaction_features(df, feature_pairs):
        """创建交互特征"""
        print("创建交互特征...")
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        print("交互特征创建完成")
        return df

def main():
    """主函数 - 数据预处理"""
    preprocessor = DataPreprocessor()
    
    # 加载数据
    df = preprocessor.load_data('data/yellow_sea_sst_data.csv')
    
    # 数据预处理流程
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.create_temporal_features(df)
    df = preprocessor.create_lag_features(df)
    df = preprocessor.create_derived_features(df)
    df = preprocessor.remove_outliers(df)
    
    # 特征工程
    engineer = FeatureEngineer()
    df = engineer.create_physics_features(df)
    
    # 交互特征
    interaction_pairs = [
        ('month_sin', 'day_of_year_sin'),
        ('season_sin', 'mean_sst_lag_7'),
        ('thermal_capacity', 'mixed_layer_depth')
    ]
    df = engineer.create_interaction_features(df, interaction_pairs)
    
    # 准备训练数据
    X_train, X_test, y_train, y_test, feature_columns = preprocessor.prepare_training_data(df)
    
    # 可视化
    preprocessor.visualize_data(df)
    
    # 保存处理后的数据
    df.to_csv('data/processed_yellow_sea_sst_data.csv', index=False)
    
    print(f"\n=== 数据预处理完成 ===")
    print(f"总特征数: {len(feature_columns)}")
    print(f"训练样本数: {len(X_train)}")
    print(f"测试样本数: {len(X_test)}")
    
    return df, X_train, X_test, y_train, y_test, feature_columns

if __name__ == "__main__":
    main()






