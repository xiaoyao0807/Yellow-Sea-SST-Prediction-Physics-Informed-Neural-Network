"""
模型训练脚本
整合数据预处理、PINN模型训练和评估的完整流程
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing import DataPreprocessor, FeatureEngineer
from pinn_model import PhysicsInformedNN, PINNTrainer, PINNPredictor, create_data_loaders
from config import *

class ModelTrainingPipeline:
    """模型训练流水线"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor(config=PREPROCESSING_CONFIG)
        self.engineer = FeatureEngineer()
        self.model = None
        self.trainer = None
        self.predictor = None
        self.feature_columns = None
        self.df_raw = None
        
    def run_complete_pipeline(self, data_file='data/yellow_sea_sst_data.csv'):
        """运行完整的训练流水线"""
        print("=" * 60)
        print("黄海海水温度预测模型 - PINN训练流水线")
        print("=" * 60)
        
        # 步骤1: 数据预处理
        print("\n步骤1: 数据预处理和特征工程")
        df = self._preprocess_data(data_file)
        
        # 步骤2: 准备训练数据
        print("\n步骤2: 准备训练数据")
        X_train, X_test, y_train, y_test = self._prepare_training_data(df)
        
        # 步骤3: 创建和训练PINN模型
        print("\n步骤3: 创建和训练PINN模型")
        self._train_pinn_model(X_train, X_test, y_train, y_test)
        
        # 步骤4: 模型评估
        print("\n步骤4: 模型评估")
        self._evaluate_model(X_test, y_test)
        
        # 步骤5: 生成预测和可视化
        print("\n步骤5: 生成预测和可视化")
        self._generate_predictions_and_plots(self.df_raw.copy(), X_test, y_test)

        # 步骤6: 持久化训练产物
        self._persist_artifacts()
        
        print("\n" + "=" * 60)
        print("训练流水线完成!")
        print("=" * 60)
        
        return self.model, self.predictor
    
    def _preprocess_data(self, data_file):
        """数据预处理"""
        print("加载和预处理数据...")
        
        # 加载数据
        df = self.preprocessor.load_data(data_file)
        
        # 数据预处理流程
        df = self.preprocessor.handle_missing_values(df)
        df = self.preprocessor.create_temporal_features(df)
        df = self.preprocessor.create_lag_features(df)
        df = self.preprocessor.create_derived_features(df)
        df = self.preprocessor.remove_outliers(df)
        
        # 特征工程（根据配置决定是否执行）
        if self.preprocessor.enable_physics_features:
            df = self.engineer.create_physics_features(df)
        
        if self.preprocessor.enable_interaction_features:
            interaction_pairs = [
                ('month_sin', 'day_of_year_sin'),
                ('season_sin', 'mean_sst_lag_7'),
                ('thermal_capacity', 'mixed_layer_depth')
            ]
            df = self.engineer.create_interaction_features(df, interaction_pairs)
        
        # 丢弃因滞后导致的缺失行
        df = df.dropna().reset_index(drop=True)
        
        # 可视化原始尺度数据
        self.preprocessor.visualize_data(df)
        
        # 准备特征列并进行缩放
        feature_columns = [col for col in df.columns if col not in ['date', 'mean_sst']]
        self.feature_columns = feature_columns
        
        self.df_raw = df.copy()
        df_scaled = self.preprocessor.scale_features(df.copy(), feature_columns)
        
        # 保存处理后的数据
        df.to_csv('data/processed_yellow_sea_sst_data_raw.csv', index=False)
        df_scaled.to_csv('data/processed_yellow_sea_sst_data.csv', index=False)
        
        print(f"数据预处理完成，总特征数: {len(feature_columns)}")
        return df_scaled
    
    def _prepare_training_data(self, df):
        """准备训练数据"""
        print("准备训练数据...")
        
        # 准备训练数据
        X_train, X_test, y_train, y_test, feature_columns = self.preprocessor.prepare_training_data(df)
        if self.feature_columns is None:
            self.feature_columns = feature_columns
        elif feature_columns != self.feature_columns:
            print("⚠️ 特征列不一致，使用预处理阶段的配置")
        
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")
        print(f"特征数: {len(feature_columns)}")
        
        return X_train, X_test, y_train, y_test
    
    def _train_pinn_model(self, X_train, X_test, y_train, y_test):
        """训练PINN模型"""
        print("创建PINN模型...")

        train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)

        scaler_params = None
        if self.feature_columns and 'mean_sst_lag_1' in self.feature_columns:
            lag1_idx = self.feature_columns.index('mean_sst_lag_1')
            scaler_params = {
                'ss_mean': float(self.preprocessor.scaler.mean_[lag1_idx]),
                'ss_scale': float(self.preprocessor.scaler.scale_[lag1_idx]),
                'mm_min': float(self.preprocessor.temperature_scaler.data_min_[0]),
                'mm_scale': float(self.preprocessor.temperature_scaler.data_range_[0]),
            }
            print(f"Scaler 参数: lag1 SS(μ={scaler_params['ss_mean']:.2f}, σ={scaler_params['ss_scale']:.2f}), "
                  f"target MM(min={scaler_params['mm_min']:.2f}, range={scaler_params['mm_scale']:.2f})")

        self.model = PhysicsInformedNN(
            input_dim=len(self.feature_columns),
            hidden_layers=MODEL_CONFIG['hidden_layers'],
            activation=MODEL_CONFIG['activation'],
            feature_names=self.feature_columns,
            scaler_params=scaler_params,
        )
        
        # 创建训练器
        self.trainer = PINNTrainer(self.model)
        
        # 训练模型
        print("开始训练...")
        train_losses, val_losses, physics_losses = self.trainer.train(train_loader, test_loader)
        
        # 绘制训练历史
        self.trainer.plot_training_history()
        
        # 创建预测器
        self.predictor = PINNPredictor(
            self.model, 
            self.preprocessor.scaler, 
            self.preprocessor.temperature_scaler
        )
        
        print("模型训练完成!")
    
    def _evaluate_model(self, X_test, y_test):
        """评估模型"""
        print("评估模型性能...")
        
        # 预测
        y_pred = self.predictor.predict(X_test)
        
        # 反归一化真实值
        y_test_original = self.preprocessor.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        # 计算MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
        
        print(f"\n=== 模型评估结果 ===")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f} °C")
        print(f"平均绝对误差 (MAE): {mae:.4f} °C")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
        
        # 保存评估结果
        evaluation_results = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        results_df = pd.DataFrame([evaluation_results])
        results_df.to_csv('results/model_evaluation.csv', index=False)
        
        return evaluation_results
    
    def _generate_predictions_and_plots(self, df, X_test, y_test):
        """生成预测和可视化"""
        print("生成预测和可视化...")
        
        # 预测
        y_pred = self.predictor.predict(X_test)
        
        # 反归一化真实值
        y_test_original = self.preprocessor.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # 创建预测结果DataFrame
        predictions_df = pd.DataFrame({
            'actual': y_test_original,
            'predicted': y_pred,
            'error': y_test_original - y_pred,
            'abs_error': np.abs(y_test_original - y_pred)
        })
        
        # 保存预测结果
        predictions_df.to_csv('results/predictions.csv', index=False)
        
        # 生成可视化
        self._create_evaluation_plots(predictions_df)
        
        # 生成未来预测
        self._generate_future_predictions(df)
    
    def _create_evaluation_plots(self, predictions_df):
        """创建评估图表"""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测vs实际
        axes[0, 0].scatter(predictions_df['actual'], predictions_df['predicted'], alpha=0.6)
        axes[0, 0].plot([predictions_df['actual'].min(), predictions_df['actual'].max()], 
                       [predictions_df['actual'].min(), predictions_df['actual'].max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('实际温度 (°C)')
        axes[0, 0].set_ylabel('预测温度 (°C)')
        axes[0, 0].set_title('预测 vs 实际')
        
        # 计算R²
        r2 = r2_score(predictions_df['actual'], predictions_df['predicted'])
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 残差图
        axes[0, 1].scatter(predictions_df['predicted'], predictions_df['error'], alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测温度 (°C)')
        axes[0, 1].set_ylabel('残差 (°C)')
        axes[0, 1].set_title('残差分析')
        
        # 3. 误差分布
        axes[1, 0].hist(predictions_df['error'], bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('预测误差 (°C)')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].set_title('误差分布')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        
        # 4. 时间序列预测
        n_points = min(100, len(predictions_df))
        indices = np.arange(n_points)
        axes[1, 1].plot(indices, predictions_df['actual'][:n_points], label='实际', alpha=0.7)
        axes[1, 1].plot(indices, predictions_df['predicted'][:n_points], label='预测', alpha=0.7)
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('温度 (°C)')
        axes[1, 1].set_title('时间序列预测对比')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('plots/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_future_predictions(self, df):
        """生成未来预测"""
        print("生成未来30天预测...")
        
        # 使用最后30天的数据作为输入
        last_data = df.tail(30)[self.feature_columns].values
        
        # 标准化
        last_data_scaled = self.preprocessor.scaler.transform(last_data)
        
        # 生成未来30天预测
        future_predictions = self.predictor.predict_sequence(
            last_data_scaled[-1], steps=30
        )
        
        # 创建未来日期
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=30, 
            freq='D'
        )
        
        # 创建预测结果DataFrame
        future_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sst': future_predictions
        })
        
        # 保存未来预测
        future_df.to_csv('results/future_predictions.csv', index=False)
        
        # 可视化未来预测
        plt.figure(figsize=(12, 6))
        
        # 历史数据
        plt.plot(df['date'].tail(60), df['mean_sst'].tail(60), 
                label='历史数据', alpha=0.7)
        
        # 未来预测
        plt.plot(future_df['date'], future_df['predicted_sst'], 
                label='未来预测', color='red', linewidth=2)
        
        plt.xlabel('日期')
        plt.ylabel('海水温度 (°C)')
        plt.title('黄海海水温度未来30天预测')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('plots/future_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"未来预测已保存到 results/future_predictions.csv")
        print(f"预测温度范围: {future_predictions.min():.2f}°C - {future_predictions.max():.2f}°C")

    def _persist_artifacts(self):
        """保存模型及预处理相关资产"""
        print("\n保存模型和预处理配置...")
        model_dir = OUTPUT_CONFIG['model_save_path']
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(
            self.preprocessor.scaler,
            os.path.join(model_dir, 'feature_scaler.pkl')
        )
        joblib.dump(
            self.preprocessor.temperature_scaler,
            os.path.join(model_dir, 'temperature_scaler.pkl')
        )

        with open(os.path.join(model_dir, 'feature_columns.json'), 'w', encoding='utf-8') as f:
            json.dump(self.feature_columns, f, ensure_ascii=False, indent=2)

        metadata = {
            'feature_columns': self.feature_columns,
            'preprocessing_config': self.preprocessor.config,
            'lag_features': self.preprocessor.lag_features if self.preprocessor.enable_lag_features else [],
            'training_data': {
                'start_date': self.df_raw['date'].min().strftime('%Y-%m-%d') if self.df_raw is not None else None,
                'end_date': self.df_raw['date'].max().strftime('%Y-%m-%d') if self.df_raw is not None else None,
                'num_samples': int(len(self.df_raw)) if self.df_raw is not None else 0
            }
        }

        with open(os.path.join(model_dir, 'training_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print("模型与配置已保存。")

def main():
    """主函数"""
    from config import resolve_training_data_csv

    data_file = resolve_training_data_csv()
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请运行 sgli_l3_sst_extract_gee.py 生成 GCOM-C 数据，或放置 data/yellow_sea_sst_data.csv")
        return
    
    # 创建训练流水线
    pipeline = ModelTrainingPipeline()
    
    # 运行完整流水线
    model, predictor = pipeline.run_complete_pipeline(data_file)
    
    print("\n训练完成! 模型和结果已保存到相应目录。")
    print("主要输出文件:")
    print("- models/best_model.pth: 训练好的模型")
    print("- results/model_evaluation.csv: 评估指标")
    print("- results/predictions.csv: 预测结果")
    print("- results/future_predictions.csv: 未来预测")
    print("- plots/: 各种可视化图表")

if __name__ == "__main__":
    main()






