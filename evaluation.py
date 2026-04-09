"""
模型评估和验证模块
生成详细的评估报告，包括统计指标、物理合理性验证和可视化分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, predictor, scaler, temperature_scaler):
        self.predictor = predictor
        self.scaler = scaler
        self.temperature_scaler = temperature_scaler
        
    def comprehensive_evaluation(self, X_test, y_test, df_original):
        """综合评估"""
        print("=" * 60)
        print("模型综合评估报告")
        print("=" * 60)
        
        # 1. 基础统计评估
        print("\n1. 基础统计评估")
        stats_results = self._statistical_evaluation(X_test, y_test)
        
        # 2. 物理合理性评估
        print("\n2. 物理合理性评估")
        physics_results = self._physics_evaluation(X_test, y_test, df_original)
        
        # 3. 季节性评估
        print("\n3. 季节性模式评估")
        seasonal_results = self._seasonal_evaluation(X_test, y_test, df_original)
        
        # 4. 误差分析
        print("\n4. 误差分析")
        error_results = self._error_analysis(X_test, y_test)
        
        # 5. 生成评估报告
        print("\n5. 生成评估报告")
        self._generate_evaluation_report(stats_results, physics_results, 
                                       seasonal_results, error_results)
        
        # 6. 创建可视化
        print("\n6. 创建评估可视化")
        self._create_evaluation_visualizations(X_test, y_test, df_original)
        
        return {
            'statistical': stats_results,
            'physics': physics_results,
            'seasonal': seasonal_results,
            'error': error_results
        }
    
    def _statistical_evaluation(self, X_test, y_test):
        """统计评估"""
        # 预测
        y_pred = self.predictor.predict(X_test)
        
        # 反归一化真实值
        y_test_original = self.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # 计算各种统计指标
        mse = mean_squared_error(y_test_original, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)
        
        # MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100
        
        # MAE (中位数绝对误差)
        median_ae = np.median(np.abs(y_test_original - y_pred))
        
        # 最大误差
        max_error = np.max(np.abs(y_test_original - y_pred))
        
        # 误差标准差
        error_std = np.std(y_test_original - y_pred)
        
        # 相关系数
        correlation = np.corrcoef(y_test_original, y_pred)[0, 1]
        
        results = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Median_AE': median_ae,
            'Max_Error': max_error,
            'Error_Std': error_std,
            'Correlation': correlation
        }
        
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"均方根误差 (RMSE): {rmse:.4f} °C")
        print(f"平均绝对误差 (MAE): {mae:.4f} °C")
        print(f"决定系数 (R²): {r2:.4f}")
        print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")
        print(f"中位数绝对误差: {median_ae:.4f} °C")
        print(f"最大误差: {max_error:.4f} °C")
        print(f"误差标准差: {error_std:.4f} °C")
        print(f"相关系数: {correlation:.4f}")
        
        return results
    
    def _physics_evaluation(self, X_test, y_test, df_original):
        """物理合理性评估"""
        y_pred = self.predictor.predict(X_test)
        y_test_original = self.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # 1. 温度范围合理性
        temp_min, temp_max = PHYSICS_CONFIG['temperature_range']
        pred_in_range = np.sum((y_pred >= temp_min) & (y_pred <= temp_max))
        range_violation_rate = 1 - (pred_in_range / len(y_pred))
        
        # 2. 温度变化率合理性
        pred_changes = np.diff(y_pred)
        actual_changes = np.diff(y_test_original)
        
        # 计算变化率的统计特性
        pred_change_std = np.std(pred_changes)
        actual_change_std = np.std(actual_changes)
        change_consistency = 1 - abs(pred_change_std - actual_change_std) / actual_change_std
        
        # 3. 季节性一致性
        seasonal_consistency = self._check_seasonal_consistency(y_pred, df_original)
        
        # 4. 空间平滑性
        spatial_smoothness = self._check_spatial_smoothness(y_pred)
        
        results = {
            'Range_Violation_Rate': range_violation_rate,
            'Change_Consistency': change_consistency,
            'Seasonal_Consistency': seasonal_consistency,
            'Spatial_Smoothness': spatial_smoothness
        }
        
        print(f"温度范围违反率: {range_violation_rate:.2%}")
        print(f"变化率一致性: {change_consistency:.4f}")
        print(f"季节性一致性: {seasonal_consistency:.4f}")
        print(f"空间平滑性: {spatial_smoothness:.4f}")
        
        return results
    
    def _check_seasonal_consistency(self, y_pred, df_original):
        """检查季节性一致性"""
        # 这里简化处理，实际应用中需要更复杂的季节性分析
        # 假设预测值应该遵循某种季节性模式
        
        # 计算预测值的季节性变化
        pred_seasonal_var = np.var(y_pred)
        
        # 计算历史数据的季节性变化
        historical_seasonal_var = np.var(df_original['mean_sst'])
        
        # 季节性一致性指标
        consistency = 1 - abs(pred_seasonal_var - historical_seasonal_var) / historical_seasonal_var
        return max(0, consistency)
    
    def _check_spatial_smoothness(self, y_pred):
        """检查空间平滑性"""
        # 计算预测值的变化率
        changes = np.diff(y_pred)
        
        # 平滑性指标：变化率的标准差越小，越平滑
        smoothness = 1 / (1 + np.std(changes))
        return smoothness
    
    def _seasonal_evaluation(self, X_test, y_test, df_original):
        """季节性评估"""
        y_pred = self.predictor.predict(X_test)
        y_test_original = self.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # 获取测试数据的时间信息
        test_dates = df_original['date'].tail(len(y_test))
        test_months = test_dates.dt.month
        
        # 按月份分析误差
        monthly_errors = {}
        monthly_r2 = {}
        
        for month in range(1, 13):
            month_mask = test_months == month
            if np.sum(month_mask) > 0:
                month_actual = y_test_original[month_mask]
                month_pred = y_pred[month_mask]
                
                monthly_errors[month] = np.mean(np.abs(month_actual - month_pred))
                monthly_r2[month] = r2_score(month_actual, month_pred)
        
        # 计算季节性性能指标
        seasonal_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
        seasonal_r2 = r2_score(y_test_original, y_pred)
        
        results = {
            'Seasonal_RMSE': seasonal_rmse,
            'Seasonal_R2': seasonal_r2,
            'Monthly_Errors': monthly_errors,
            'Monthly_R2': monthly_r2
        }
        
        print(f"季节性RMSE: {seasonal_rmse:.4f} °C")
        print(f"季节性R²: {seasonal_r2:.4f}")
        
        return results
    
    def _error_analysis(self, X_test, y_test):
        """误差分析"""
        y_pred = self.predictor.predict(X_test)
        y_test_original = self.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        errors = y_test_original - y_pred
        abs_errors = np.abs(errors)
        
        # 误差分布分析
        error_percentiles = np.percentile(abs_errors, [25, 50, 75, 90, 95, 99])
        
        # 误差趋势分析
        error_trend = np.polyfit(range(len(errors)), errors, 1)[0]
        
        # 误差自相关
        error_autocorr = np.corrcoef(errors[:-1], errors[1:])[0, 1]
        
        results = {
            'Error_Percentiles': error_percentiles,
            'Error_Trend': error_trend,
            'Error_Autocorr': error_autocorr,
            'Mean_Error': np.mean(errors),
            'Error_Skewness': self._calculate_skewness(errors),
            'Error_Kurtosis': self._calculate_kurtosis(errors)
        }
        
        print(f"误差25%分位数: {error_percentiles[0]:.4f} °C")
        print(f"误差中位数: {error_percentiles[1]:.4f} °C")
        print(f"误差75%分位数: {error_percentiles[2]:.4f} °C")
        print(f"误差90%分位数: {error_percentiles[3]:.4f} °C")
        print(f"误差95%分位数: {error_percentiles[4]:.4f} °C")
        print(f"误差99%分位数: {error_percentiles[5]:.4f} °C")
        print(f"误差趋势: {error_trend:.6f}")
        print(f"误差自相关: {error_autocorr:.4f}")
        print(f"平均误差: {np.mean(errors):.4f} °C")
        
        return results
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _generate_evaluation_report(self, stats_results, physics_results, 
                                  seasonal_results, error_results):
        """生成评估报告"""
        report = {
            'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'Physics-Informed Neural Network (PINN)',
            'target': '黄海海水温度预测',
            'statistical_metrics': stats_results,
            'physics_metrics': physics_results,
            'seasonal_metrics': seasonal_results,
            'error_analysis': error_results
        }
        
        # 保存报告
        report_df = pd.DataFrame([report])
        report_df.to_json('results/comprehensive_evaluation_report.json', 
                         orient='records', indent=2)
        
        # 生成Markdown报告
        self._generate_markdown_report(report)
        
        print("评估报告已生成:")
        print("- results/comprehensive_evaluation_report.json")
        print("- results/evaluation_report.md")
    
    def _generate_markdown_report(self, report):
        """生成Markdown格式的评估报告"""
        md_content = f"""# 黄海海水温度预测模型评估报告

## 基本信息
- **评估日期**: {report['evaluation_date']}
- **模型类型**: {report['model_type']}
- **预测目标**: {report['target']}

## 统计性能指标

### 基础指标
- **均方误差 (MSE)**: {report['statistical_metrics']['MSE']:.4f}
- **均方根误差 (RMSE)**: {report['statistical_metrics']['RMSE']:.4f} °C
- **平均绝对误差 (MAE)**: {report['statistical_metrics']['MAE']:.4f} °C
- **决定系数 (R²)**: {report['statistical_metrics']['R2']:.4f}
- **平均绝对百分比误差 (MAPE)**: {report['statistical_metrics']['MAPE']:.2f}%

### 高级指标
- **中位数绝对误差**: {report['statistical_metrics']['Median_AE']:.4f} °C
- **最大误差**: {report['statistical_metrics']['Max_Error']:.4f} °C
- **误差标准差**: {report['statistical_metrics']['Error_Std']:.4f} °C
- **相关系数**: {report['statistical_metrics']['Correlation']:.4f}

## 物理合理性评估

- **温度范围违反率**: {report['physics_metrics']['Range_Violation_Rate']:.2%}
- **变化率一致性**: {report['physics_metrics']['Change_Consistency']:.4f}
- **季节性一致性**: {report['physics_metrics']['Seasonal_Consistency']:.4f}
- **空间平滑性**: {report['physics_metrics']['Spatial_Smoothness']:.4f}

## 季节性性能

- **季节性RMSE**: {report['seasonal_metrics']['Seasonal_RMSE']:.4f} °C
- **季节性R²**: {report['seasonal_metrics']['Seasonal_R2']:.4f}

## 误差分析

### 误差分布
- **25%分位数**: {report['error_analysis']['Error_Percentiles'][0]:.4f} °C
- **中位数**: {report['error_analysis']['Error_Percentiles'][1]:.4f} °C
- **75%分位数**: {report['error_analysis']['Error_Percentiles'][2]:.4f} °C
- **90%分位数**: {report['error_analysis']['Error_Percentiles'][3]:.4f} °C
- **95%分位数**: {report['error_analysis']['Error_Percentiles'][4]:.4f} °C
- **99%分位数**: {report['error_analysis']['Error_Percentiles'][5]:.4f} °C

### 误差特性
- **平均误差**: {report['error_analysis']['Mean_Error']:.4f} °C
- **误差趋势**: {report['error_analysis']['Error_Trend']:.6f}
- **误差自相关**: {report['error_analysis']['Error_Autocorr']:.4f}
- **误差偏度**: {report['error_analysis']['Error_Skewness']:.4f}
- **误差峰度**: {report['error_analysis']['Error_Kurtosis']:.4f}

## 模型评估结论

基于以上评估指标，该PINN模型在黄海海水温度预测任务中表现{'良好' if report['statistical_metrics']['R2'] > 0.8 else '一般' if report['statistical_metrics']['R2'] > 0.6 else '较差'}。

### 优势
- R²值达到 {report['statistical_metrics']['R2']:.3f}，表明模型能够解释大部分数据变异
- 物理约束有效，温度范围违反率仅为 {report['physics_metrics']['Range_Violation_Rate']:.2%}

### 改进建议
- 如果R²值低于0.8，建议增加更多特征或调整模型结构
- 如果物理违反率较高，建议加强物理约束
- 如果季节性一致性较低，建议优化时间特征工程

---
*报告生成时间: {report['evaluation_date']}*
"""
        
        with open('results/evaluation_report.md', 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _create_evaluation_visualizations(self, X_test, y_test, df_original):
        """创建评估可视化"""
        y_pred = self.predictor.predict(X_test)
        y_test_original = self.temperature_scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()
        
        # 创建综合评估图表
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. 预测vs实际散点图
        axes[0, 0].scatter(y_test_original, y_pred, alpha=0.6, s=20)
        axes[0, 0].plot([y_test_original.min(), y_test_original.max()], 
                       [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('实际温度 (°C)')
        axes[0, 0].set_ylabel('预测温度 (°C)')
        axes[0, 0].set_title('预测 vs 实际')
        
        # 2. 残差图
        residuals = y_test_original - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=20)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('预测温度 (°C)')
        axes[0, 1].set_ylabel('残差 (°C)')
        axes[0, 1].set_title('残差分析')
        
        # 3. 误差分布直方图
        axes[0, 2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('残差 (°C)')
        axes[0, 2].set_ylabel('频次')
        axes[0, 2].set_title('残差分布')
        axes[0, 2].axvline(x=0, color='r', linestyle='--')
        
        # 4. Q-Q图
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q图 (正态性检验)')
        
        # 5. 时间序列对比
        n_points = min(100, len(y_test_original))
        indices = np.arange(n_points)
        axes[1, 1].plot(indices, y_test_original[:n_points], label='实际', alpha=0.7)
        axes[1, 1].plot(indices, y_pred[:n_points], label='预测', alpha=0.7)
        axes[1, 1].set_xlabel('样本索引')
        axes[1, 1].set_ylabel('温度 (°C)')
        axes[1, 1].set_title('时间序列对比')
        axes[1, 1].legend()
        
        # 6. 误差随时间变化
        axes[1, 2].plot(indices, np.abs(residuals[:n_points]), alpha=0.7)
        axes[1, 2].set_xlabel('样本索引')
        axes[1, 2].set_ylabel('绝对误差 (°C)')
        axes[1, 2].set_title('绝对误差随时间变化')
        
        # 7. 月度性能
        test_dates = df_original['date'].tail(len(y_test))
        monthly_mae = []
        months = []
        for month in range(1, 13):
            month_mask = test_dates.dt.month == month
            if np.sum(month_mask) > 0:
                monthly_mae.append(np.mean(np.abs(residuals[month_mask])))
                months.append(month)
        
        if months:
            axes[2, 0].bar(months, monthly_mae, alpha=0.7)
            axes[2, 0].set_xlabel('月份')
            axes[2, 0].set_ylabel('平均绝对误差 (°C)')
            axes[2, 0].set_title('月度性能')
        
        # 8. 误差箱线图
        axes[2, 1].boxplot([residuals], labels=['残差'])
        axes[2, 1].set_ylabel('残差 (°C)')
        axes[2, 1].set_title('误差箱线图')
        
        # 9. 累积误差
        sorted_errors = np.sort(np.abs(residuals))
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[2, 2].plot(sorted_errors, cumulative_prob)
        axes[2, 2].set_xlabel('绝对误差 (°C)')
        axes[2, 2].set_ylabel('累积概率')
        axes[2, 2].set_title('累积误差分布')
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """主函数 - 模型评估"""
    print("模型评估模块")
    print("请先运行 training.py 训练模型，然后使用此模块进行评估")

if __name__ == "__main__":
    main()










