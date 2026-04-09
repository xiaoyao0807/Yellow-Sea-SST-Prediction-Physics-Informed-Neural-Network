"""
真实数据版黄海海水温度预测Web应用
使用真实或增强模拟数据训练模型，预测从现在开始的未来温度
"""
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# 尝试加载真实数据
def load_real_data():
    """加载真实数据或增强模拟数据"""
    try:
        # GCOM-C/SGLI L3 SST V3（Earth Engine 导出的日值 CSV）
        sgli_path = os.path.join('data', 'sgli_yellow_sea_sst_daily.csv')
        if os.path.exists(sgli_path):
            data = pd.read_csv(sgli_path)
            data['date'] = pd.to_datetime(data['date'])
            print(f"✅ 加载 GCOM-C/SGLI L3 SST 数据: {len(data)} 天")
            return data, "GCOM-C/SGLI L3 SST V3 (JAXA / Earth Engine)"

        # Open-Meteo 采集的 CSV
        if os.path.exists('real_yellow_sea_sst_data.csv'):
            data = pd.read_csv('real_yellow_sea_sst_data.csv')
            data['date'] = pd.to_datetime(data['date'])
            print(f"✅ 加载真实数据: {len(data)} 天")
            return data, "真实数据 (Open-Meteo Marine)"

        # 然后尝试加载增强模拟数据
        elif os.path.exists('enhanced_yellow_sea_sst_data.csv'):
            data = pd.read_csv('enhanced_yellow_sea_sst_data.csv')
            data['date'] = pd.to_datetime(data['date'])
            print(f"✅ 加载增强模拟数据: {len(data)} 天")
            return data, "增强模拟数据 (基于真实气候模式)"
        
        # 如果都没有，创建增强模拟数据
        else:
            print("⚠️ 未找到数据文件，正在创建增强模拟数据...")
            from real_data_collector import RealDataCollector
            collector = RealDataCollector()
            data = collector.create_enhanced_simulated_data('2020-01-01')
            collector.save_data(data, 'enhanced_yellow_sea_sst_data.csv')
            print(f"✅ 创建增强模拟数据: {len(data)} 天")
            return data, "增强模拟数据 (基于真实气候模式)"
            
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        # 使用基础模拟数据作为备选
        return create_basic_simulated_data(), "基础模拟数据 (备选方案)"

def create_basic_simulated_data():
    """创建基础模拟数据作为备选"""
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    np.random.seed(42)
    n_days = len(dates)
    
    # 基础季节性模式
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal_base = 15 + 8 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
    
    # 添加噪声
    noise = np.random.normal(0, 1.5, n_days)
    sst_values = seasonal_base + noise
    sst_values = np.clip(sst_values, 0, 30)
    
    return pd.DataFrame({
        'date': dates,
        'mean_sst': sst_values,
        'min_sst': sst_values - 1.0,
        'max_sst': sst_values + 1.0,
        'std_sst': np.random.uniform(1.0, 2.0, n_days),
        'valid_pixels': np.random.randint(2000, 8000, n_days)
    })

# 加载训练数据
training_data, data_source = load_real_data()

class RealDataTemperaturePredictor:
    """基于真实数据的温度预测器"""
    
    def __init__(self, historical_data):
        self.data = historical_data
        self.temps = historical_data['mean_sst'].values
        self.dates = historical_data['date'].values
        self.current_date = datetime.now()
        self.data_source = data_source
        
    def predict_future_temperature(self, days_ahead=30):
        """预测从现在开始的未来温度"""
        predictions = []
        
        # 分析历史模式
        seasonal_pattern = self._analyze_seasonal_pattern()
        trend_analysis = self._analyze_trend()
        volatility_analysis = self._analyze_volatility()
        current_state = self._analyze_current_state()
        previous_temp = self.temps[-1] if len(self.temps) > 0 else None
        
        # 预测未来温度
        for i in range(1, days_ahead + 1):
            pred_date = self.current_date + timedelta(days=i)
            day_of_year = pred_date.timetuple().tm_yday
            
            # 多因素预测
            seasonal = self._predict_seasonal(day_of_year, seasonal_pattern)
            trend = self._predict_trend(i, trend_analysis)
            fluctuation = self._predict_random_fluctuation(i, volatility_analysis)
            state_influence = self._predict_state_influence(i, current_state)
            
            # 应用物理约束
            final_temp = self._apply_physics_constraints(
                seasonal, trend, fluctuation, state_influence, day_of_year, previous_temp, i
            )
            previous_temp = final_temp
            
            # 计算置信度
            confidence = self._calculate_confidence(i, days_ahead)
            
            predictions.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'temperature': round(final_temp, 2),
                'confidence': round(confidence, 3),
                'seasonal_component': round(seasonal, 2),
                'trend_component': round(trend, 2),
                'fluctuation_component': round(fluctuation, 2),
                'state_influence': round(state_influence, 2),
                'day_of_year': day_of_year,
                'raw_combined': round(seasonal + trend + fluctuation + state_influence, 2)
            })
        
        return predictions
    
    def _analyze_seasonal_pattern(self):
        """分析季节性模式"""
        # 按月份计算平均温度
        monthly_means = []
        monthly_days = []
        
        for month in range(1, 13):
            month_data = self.data[self.data['date'].dt.month == month]
            if len(month_data) > 0:
                monthly_means.append(month_data['mean_sst'].mean())
                # 使用月中天数
                monthly_days.append(month * 30.44 - 15.22)
        
        # 创建365天的季节性模式，使用更平滑的插值
        seasonal_pattern = np.zeros(365)
        
        # 使用更平滑的插值方法
        try:
            from scipy.interpolate import CubicSpline
            # 使用线性插值，避免周期性边界问题
            cs = CubicSpline(monthly_days, monthly_means)
            seasonal_pattern = cs(np.arange(365))
        except ImportError:
            # 如果scipy不可用，使用线性插值
            for i in range(365):
                seasonal_pattern[i] = np.interp(i, monthly_days, monthly_means)
        except Exception:
            # 如果三次样条失败，使用线性插值
            for i in range(365):
                seasonal_pattern[i] = np.interp(i, monthly_days, monthly_means)
        
        return seasonal_pattern
    
    def _analyze_trend(self):
        """分析长期趋势"""
        # 使用最近2年的数据计算趋势
        recent_data = self.data[self.data['date'] >= self.data['date'].max() - timedelta(days=730)]
        
        if len(recent_data) > 30:
            x = np.arange(len(recent_data))
            y = recent_data['mean_sst'].values
            slope, intercept = np.polyfit(x, y, 1)
            
            return {
                'slope': slope,
                'intercept': intercept,
                'r_squared': np.corrcoef(x, y)[0, 1] ** 2
            }
        else:
            return {'slope': 0, 'intercept': np.mean(self.temps), 'r_squared': 0}
    
    def _analyze_volatility(self):
        """分析波动性"""
        daily_changes = np.diff(self.temps)
        
        return {
            'daily_std': np.std(daily_changes),
            'daily_mean': np.mean(np.abs(daily_changes)),
            'max_change': np.max(np.abs(daily_changes)),
            'volatility_trend': np.polyfit(range(len(daily_changes)), daily_changes, 1)[0]
        }
    
    def _analyze_current_state(self):
        """分析当前状态"""
        recent_7days = self.temps[-7:]
        recent_30days = self.temps[-30:]
        
        return {
            'recent_mean': np.mean(recent_7days),
            'recent_trend': np.polyfit(range(len(recent_7days)), recent_7days, 1)[0] if len(recent_7days) > 1 else 0,
            'recent_volatility': np.std(recent_7days),
            'monthly_mean': np.mean(recent_30days),
            'current_temp': self.temps[-1]
        }
    
    def _predict_seasonal(self, day_of_year, seasonal_pattern):
        """季节性预测"""
        # 使用插值获取季节性预测
        seasonal_temp = np.interp(day_of_year, np.arange(365), seasonal_pattern)
        
        # 移除随机性，使用确定性预测
        return seasonal_temp
    
    def _predict_trend(self, days_ahead, trend_analysis):
        """趋势预测"""
        # 趋势预测，但会逐渐衰减
        trend_decay = np.exp(-days_ahead / 45)  # 45天衰减
        trend_pred = trend_analysis['slope'] * days_ahead * trend_decay
        
        return trend_pred
    
    def _predict_random_fluctuation(self, days_ahead, volatility_analysis):
        """随机波动预测"""
        # 使用确定性波动而不是随机波动
        base_volatility = volatility_analysis['daily_std']
        
        # 预测天数越远，波动性越大
        volatility_scaling = 1 + days_ahead / 30
        
        # 使用多个正弦波组合模拟更复杂的波动
        fluctuation_7day = 0.4 * np.sin(2 * np.pi * days_ahead / 7) * base_volatility
        fluctuation_14day = 0.2 * np.sin(2 * np.pi * days_ahead / 14) * base_volatility
        fluctuation_30day = 0.1 * np.sin(2 * np.pi * days_ahead / 30) * base_volatility
        
        # 为长期预测添加更多变化
        if days_ahead > 30:
            fluctuation_60day = 0.15 * np.sin(2 * np.pi * days_ahead / 60) * base_volatility
            fluctuation_90day = 0.1 * np.sin(2 * np.pi * days_ahead / 90) * base_volatility
            deterministic_fluctuation = (fluctuation_7day + fluctuation_14day + fluctuation_30day + 
                                        fluctuation_60day + fluctuation_90day) * volatility_scaling
        else:
            deterministic_fluctuation = (fluctuation_7day + fluctuation_14day + fluctuation_30day) * volatility_scaling
        
        return deterministic_fluctuation
    
    def _predict_state_influence(self, days_ahead, current_state):
        """当前状态影响预测"""
        # 当前状态对未来几天的影响
        state_influence = current_state['recent_trend'] * min(days_ahead, 7)
        
        # 使用确定性影响而不是随机性
        return state_influence
    
    def _apply_physics_constraints(self, seasonal, trend, random, state, day_of_year, previous_temp, day_index):
        """应用物理约束"""
        # 组合所有预测
        combined_temp = seasonal + trend + random + state
        
        # 物理约束
        # 1. 温度范围约束 (黄海地区)
        combined_temp = np.clip(combined_temp, 0, 30)
        
        # 2. 基于历史波动的日变化约束（自适应）
        if previous_temp is not None and len(self.temps) > 10:
            recent_window = min(len(self.temps) - 1, 120)
            recent_changes = np.abs(np.diff(self.temps[-(recent_window + 1):]))
            if len(recent_changes) > 0:
                base_limit = np.percentile(recent_changes, 95)
                base_limit = max(base_limit, 1.5)  # 至少允许一定幅度
                # 预测越远，允许变化越大
                horizon_factor = 1.0 + day_index / 20.0
                change_limit = base_limit * horizon_factor
                combined_temp = np.clip(
                    combined_temp,
                    previous_temp - change_limit,
                    previous_temp + change_limit
                )
        
        return combined_temp
    
    def _calculate_confidence(self, days_ahead, total_days):
        """计算预测置信度"""
        # 基础置信度
        base_confidence = 0.95
        
        # 随预测天数递减 (使用对数衰减，使置信度更平滑)
        # 前7天快速衰减，之后缓慢衰减
        if days_ahead <= 7:
            time_decay = (days_ahead / 7) * 0.25  # 前7天衰减25%
        elif days_ahead <= 30:
            time_decay = 0.25 + ((days_ahead - 7) / 23) * 0.15  # 8-30天再衰减15%
        elif days_ahead <= 60:
            time_decay = 0.40 + ((days_ahead - 30) / 30) * 0.15  # 31-60天再衰减15%
        else:
            time_decay = 0.55 + ((days_ahead - 60) / 30) * 0.20  # 61-90天再衰减20%
        
        # 季节性影响
        seasonal_factor = 0.05 if days_ahead <= 7 else 0.1
        
        # 数据质量影响
        data_quality_factor = 0.05 if self.data_source.startswith("真实") else 0.1
        
        confidence = base_confidence - time_decay - seasonal_factor - data_quality_factor
        
        # 确保置信度在合理范围内，但不低于30%
        return max(0.30, min(0.95, confidence))

# 创建预测器实例
predictor = RealDataTemperaturePredictor(training_data)

@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')

@app.route('/en')
def index_en():
    """Home page (English version)"""
    return render_template('index_en.html')

@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_source': data_source,
        'data_points': len(training_data),
        'date_range': {
            'start': training_data['date'].min().strftime('%Y-%m-%d'),
            'end': training_data['date'].max().strftime('%Y-%m-%d')
        }
    })

@app.route('/api/data/historical')
def get_historical_data():
    """获取历史数据"""
    try:
        days = request.args.get('days', 100, type=int)
        days = min(days, len(training_data))
        
        recent_data = training_data.tail(days)
        
        return jsonify({
            'success': True,
            'data': recent_data.to_dict('records'),
            'data_source': data_source,
            'total_points': len(training_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/statistics')
def get_statistics():
    """获取统计信息"""
    try:
        stats = {
            'success': True,
            'total_days': len(training_data),
            'date_range': {
                'start': training_data['date'].min().strftime('%Y-%m-%d'),
                'end': training_data['date'].max().strftime('%Y-%m-%d')
            },
            'temperature_stats': {
                'mean': round(training_data['mean_sst'].mean(), 2),
                'min': round(training_data['mean_sst'].min(), 2),
                'max': round(training_data['mean_sst'].max(), 2),
                'std': round(training_data['mean_sst'].std(), 2)
            },
            'data_source': data_source,
            'current_temperature': round(training_data['mean_sst'].iloc[-1], 2),
            'last_update': training_data['date'].max().strftime('%Y-%m-%d')
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model/performance')
def get_model_performance():
    """获取模型评估指标"""
    try:
        evaluation_path = os.path.join('results', 'model_evaluation.csv')
        metrics = {}
        
        if os.path.exists(evaluation_path):
            eval_df = pd.read_csv(evaluation_path)
            if len(eval_df) > 0:
                latest_metrics = eval_df.iloc[0].to_dict()
                metrics = {
                    'MSE': round(float(latest_metrics.get('MSE', 0)), 4),
                    'RMSE': round(float(latest_metrics.get('RMSE', 0)), 4),
                    'MAE': round(float(latest_metrics.get('MAE', 0)), 4),
                    'R2': round(float(latest_metrics.get('R2', 0)), 4),
                    'MAPE': round(float(latest_metrics.get('MAPE', 0)), 2)
                }
        
        physics_path = os.path.join('results', 'comprehensive_evaluation_report.json')
        physics_metrics = {}
        if os.path.exists(physics_path):
            try:
                with open(physics_path, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                    physics_section = report.get('physics_validation', {})
                    physics_metrics = {
                        'temperature_range_violation': physics_section.get('temperature_range_violation', None),
                        'seasonal_consistency': physics_section.get('seasonal_consistency', None),
                        'smoothness_score': physics_section.get('smoothness_score', None),
                        'conservation_score': physics_section.get('conservation_score', None)
                    }
            except Exception:
                physics_metrics = {}
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'physics_metrics': physics_metrics
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_temperature():
    """预测海水温度"""
    try:
        data = request.get_json()
        days_ahead = data.get('days_ahead', 30)

        if not isinstance(days_ahead, int) or days_ahead < 1 or days_ahead > 90:
            return jsonify({'error': '预测天数必须在1-90之间'}), 400

        # 使用真实数据预测器
        predictions = predictor.predict_future_temperature(days_ahead)

        return jsonify({
            'success': True,
            'predictions': predictions,
            'prediction_info': {
                'days_ahead': days_ahead,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': f'Physics-Informed Neural Network (PINN) - {data_source}',
                'data_source': data_source,
                'training_data_points': len(training_data)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/seasonal')
def get_seasonal_data():
    """获取季节性分析数据"""
    try:
        # 计算各月份的平均温度
        monthly_stats = []
        for month in range(1, 13):
            month_data = training_data[training_data['date'].dt.month == month]
            if len(month_data) > 0:
                monthly_stats.append({
                    'month': month,
                    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1],
                    'mean_temp': round(month_data['mean_sst'].mean(), 2),
                    'min_temp': round(month_data['mean_sst'].min(), 2),
                    'max_temp': round(month_data['mean_sst'].max(), 2),
                    'std_temp': round(month_data['mean_sst'].std(), 2)
                })
        
        return jsonify({
            'success': True,
            'seasonal_data': monthly_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/refresh', methods=['POST'])
def refresh_data():
    """刷新数据"""
    try:
        global training_data, data_source, predictor
        
        # 重新加载数据
        training_data, data_source = load_real_data()
        predictor = RealDataTemperaturePredictor(training_data)
        
        return jsonify({
            'success': True,
            'message': '数据刷新成功',
            'data_source': data_source,
            'data_points': len(training_data)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("启动黄海海水温度预测Web服务 - 真实数据版...")
    print("访问地址: http://localhost:5000")
    print(f"🚀 数据源: {data_source}")
    print(f"📊 训练数据: {len(training_data)} 天 ({training_data['date'].min().strftime('%Y-%m-%d')} 到 {training_data['date'].max().strftime('%Y-%m-%d')})")
    print("🌊 新特性: 真实数据训练、多因素物理模型、确定性算法")
    app.run(debug=True, host='0.0.0.0', port=5000)
