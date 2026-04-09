"""
真实数据收集和处理脚本
使用开放的海洋气象数据API (Open-Meteo Marine Archive) 获取黄海真实SST数据
"""
import logging
import os
from datetime import datetime, timedelta
from itertools import product
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import requests

from config import (
    DATA_CONFIG,
    OPEN_METEO_CONFIG,
    YELLOW_SEA_BOUNDS
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """使用开放数据源收集黄海SST数据"""

    def __init__(
        self,
        bounds: dict = None,
        data_config: dict = None,
        api_config: dict = None,
        session: Optional[requests.Session] = None
    ):
        self.bounds = bounds or YELLOW_SEA_BOUNDS
        self.data_config = data_config or DATA_CONFIG
        self.api_config = api_config or OPEN_METEO_CONFIG
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # 数据收集主流程
    # ------------------------------------------------------------------ #
    def collect_real_sst_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """从Open-Meteo Marine Archive获取真实SST数据"""

        start_date = start_date or self.data_config['start_date']
        end_date = end_date or self.data_config['end_date']

        logger.info(
            "开始通过 Open-Meteo 获取黄海海表温度: %s 至 %s",
            start_date,
            end_date
        )

        grid_points = self._generate_sampling_grid(
            self.api_config['grid_lat_points'],
            self.api_config['grid_lon_points']
        )

        logger.info("采样网格点数量: %d", len(grid_points))

        point_frames: List[pd.DataFrame] = []
        for idx, (lat, lon) in enumerate(grid_points):
            logger.info("采样点 %d/%d (lat=%.3f, lon=%.3f)", idx + 1, len(grid_points), lat, lon)
            df_point = self._fetch_point_timeseries(lat, lon, start_date, end_date)
            if df_point is not None and not df_point.empty:
                df_point['point_id'] = idx
                point_frames.append(df_point)
            else:
                logger.warning("采样点 lat=%.3f, lon=%.3f 未返回有效数据", lat, lon)

        if not point_frames:
            logger.error("所有采样点均未获取到数据，返回None")
            return None

        combined = pd.concat(point_frames, ignore_index=True)
        combined['date'] = pd.to_datetime(combined['date'])

        # 以日期聚合所有采样点，计算均值/标准差
        grouped = combined.groupby('date')
        result = pd.DataFrame({
            'mean_sst': grouped['mean_sst'].mean(),
            'min_sst': grouped['min_sst'].mean(),
            'max_sst': grouped['max_sst'].mean(),
            'std_sst': grouped['mean_sst'].std().fillna(0.0),
            'valid_points': grouped['mean_sst'].count()
        }).reset_index().sort_values('date')

        logger.info("成功获取 %d 天的真实SST数据", len(result))
        return result

    # ------------------------------------------------------------------ #
    # 内部工具函数
    # ------------------------------------------------------------------ #
    def _generate_sampling_grid(self, lat_points: int, lon_points: int) -> List[Tuple[float, float]]:
        """在目标范围内生成均匀采样网格"""
        latitudes = np.linspace(self.bounds['south'], self.bounds['north'], lat_points)
        longitudes = np.linspace(self.bounds['west'], self.bounds['east'], lon_points)
        return list(product(latitudes, longitudes))

    def _fetch_point_timeseries(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """按时间分块请求单个采样点"""

        chunks = list(self._chunk_date_range(start_date, end_date, self.api_config['request_chunk_days']))
        frames: List[pd.DataFrame] = []

        for chunk_start, chunk_end in chunks:
            df_chunk = self._request_marine_api(lat, lon, chunk_start, chunk_end)
            if df_chunk is not None and not df_chunk.empty:
                frames.append(df_chunk)

        if not frames:
            return None

        data = pd.concat(frames, ignore_index=True)
        data = data.drop_duplicates(subset='date').reset_index(drop=True)
        return data

    def _chunk_date_range(
        self,
        start_date: str,
        end_date: str,
        chunk_days: int
    ):
        """将时间区间按固定天数切块"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        current = start
        delta = timedelta(days=chunk_days - 1)

        while current <= end:
            chunk_end = min(current + delta, end)
            yield current.strftime('%Y-%m-%d'), chunk_end.strftime('%Y-%m-%d')
            current = chunk_end + timedelta(days=1)

    def _request_marine_api(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        retries: int = 3,
        timeout: int = 30
    ) -> Optional[pd.DataFrame]:
        """调用Open-Meteo Marine API获取单个时间段的数据"""
        params = {
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'start_date': start_date,
            'end_date': end_date,
            'daily': ','.join(self.api_config['daily_variables']),
            'timezone': 'GMT'
        }

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(
                    self.api_config['base_url'],
                    params=params,
                    timeout=timeout
                )
                response.raise_for_status()
                data = response.json()
                return self._parse_marine_payload(data)
            except Exception as exc:
                logger.warning(
                    "Open-Meteo 请求失败 (attempt %d/%d): %s",
                    attempt, retries, exc
                )

        logger.error(
            "Open-Meteo 请求在 %d 次尝试后失败 (lat=%.3f, lon=%.3f, %s-%s)",
            retries, lat, lon, start_date, end_date
        )
        return None

    @staticmethod
    def _parse_marine_payload(payload: dict) -> Optional[pd.DataFrame]:
        """解析Open-Meteo返回的数据"""
        daily = payload.get('daily')
        if not daily:
            return None

        time_series = daily.get('time')
        mean_series = daily.get('sea_surface_temperature_mean')
        min_series = daily.get('sea_surface_temperature_min')
        max_series = daily.get('sea_surface_temperature_max')

        if not (time_series and mean_series):
            return None

        df = pd.DataFrame({
            'date': time_series,
            'mean_sst': mean_series,
            'min_sst': min_series if min_series else mean_series,
            'max_sst': max_series if max_series else mean_series
        })

        # 转换为浮点数并过滤异常
        df['mean_sst'] = pd.to_numeric(df['mean_sst'], errors='coerce')
        df['min_sst'] = pd.to_numeric(df['min_sst'], errors='coerce')
        df['max_sst'] = pd.to_numeric(df['max_sst'], errors='coerce')

        df = df.dropna()
        return df

    # ------------------------------------------------------------------ #
    # 兼容旧流程的模拟数据与读写工具
    # ------------------------------------------------------------------ #
    def create_enhanced_simulated_data(self, start_date='2020-01-01', end_date=None):
        """创建增强的模拟数据（基于真实黄海气候模式）"""
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        logger.info(f"创建增强模拟数据: {len(dates)} 天")
        
        np.random.seed(42)  # 保证可重复性
        n_days = len(dates)
        
        # 1. 基于黄海真实气候的季节性模式
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # 黄海地区真实季节性模式 (基于历史观测数据)
        # 冬季最低约5°C，夏季最高约25°C
        seasonal_base = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi/2)
        
        # 2. 月内变化
        monthly_variation = 1.5 * np.sin(2 * np.pi * day_of_year / 30.44)
        
        # 3. 年际变化
        yearly_variations = []
        for year in range(start_date.year, end_date.year + 1):
            year_start = (datetime(year, 1, 1) - start_date).days
            year_end = min((datetime(year + 1, 1, 1) - start_date).days, n_days)
            year_length = year_end - year_start
            
            if year_length > 0:
                # 每年有不同的温度偏移
                year_offset = np.random.normal(0, 1.0)
                yearly_variations.extend([year_offset] * year_length)
        
        yearly_variations = np.array(yearly_variations[:n_days])
        
        # 4. 天气系统影响
        weather_noise = np.random.normal(0, 0.8, n_days)
        
        # 添加持续性天气事件
        for i in range(10, n_days-10):
            if np.random.random() < 0.1:  # 10%概率出现持续天气事件
                duration = np.random.randint(3, 15)
                intensity = np.random.uniform(-2.0, 2.0)
                # 天气事件的影响逐渐衰减
                decay_factor = np.exp(-np.arange(duration) / 4)
                weather_noise[i:i+duration] += intensity * decay_factor
        
        # 5. 长期气候变化趋势
        climate_trend = 0.02 * np.arange(n_days) / 365.25  # 缓慢升温
        
        # 6. 海洋环流影响
        circulation_effect = 1.0 * np.sin(2 * np.pi * np.arange(n_days) / (365.25 * 1.8))
        
        # 7. 厄尔尼诺/拉尼娜效应
        enso_events = np.zeros(n_days)
        for i in range(0, n_days, 365):  # 每年检查一次
            if np.random.random() < 0.25:  # 25%概率出现ENSO事件
                event_start = i + np.random.randint(0, 200)
                event_duration = np.random.randint(180, 450)
                event_intensity = np.random.uniform(-1.5, 1.5)
                
                if event_start + event_duration < n_days:
                    event_range = np.arange(event_duration)
                    enso_decay = np.exp(-event_range / 100)
                    enso_events[event_start:event_start+event_duration] += event_intensity * enso_decay
        
        # 8. 日变化
        daily_variation = 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 1)
        
        # 组合所有因素
        sst_values = (seasonal_base + 
                     monthly_variation + 
                     yearly_variations + 
                     weather_noise + 
                     climate_trend + 
                     circulation_effect + 
                     enso_events + 
                     daily_variation)
        
        # 添加一些异常值 (极端天气事件)
        for i in range(0, n_days, 120):
            if np.random.random() < 0.08:  # 8%概率出现极端事件
                extreme_intensity = np.random.uniform(-4, 4)
                extreme_duration = np.random.randint(1, 7)
                sst_values[i:i+extreme_duration] += extreme_intensity
                sst_values[i:i+extreme_duration] = np.clip(sst_values[i:i+extreme_duration], 0, 30)
        
        # 确保温度在合理范围内
        sst_values = np.clip(sst_values, 0, 30)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'mean_sst': sst_values,
            'min_sst': sst_values - np.abs(daily_variation) - np.random.uniform(0.8, 1.5, n_days),
            'max_sst': sst_values + np.abs(daily_variation) + np.random.uniform(0.8, 1.5, n_days),
            'std_sst': np.random.uniform(1.0, 2.0, n_days),
            'valid_pixels': np.random.randint(2000, 8000, n_days),
            'seasonal_component': seasonal_base,
            'weather_component': weather_noise,
            'trend_component': climate_trend,
            'circulation_component': circulation_effect,
            'enso_component': enso_events
        })
        
        # 确保min/max在合理范围内
        df['min_sst'] = np.clip(df['min_sst'], 0, df['mean_sst'])
        df['max_sst'] = np.clip(df['max_sst'], df['mean_sst'], 30)
        
        logger.info(f"增强模拟数据创建完成: {len(df)} 天")
        return df
    
    def save_data(self, df, filename='real_yellow_sea_sst_data.csv', format='csv'):
        """
        保存数据到文件
        
        Args:
            df: 数据DataFrame
            filename: 文件名
            format: 保存格式 ('csv' 或 'netcdf' 或 'both')
        """
        if df is None or len(df) == 0:
            return False
        
        success = False
        csv_filename = None
        nc_filename = None
        
        # 确定文件名
        if filename.endswith('.csv'):
            csv_filename = filename
            nc_filename = filename.replace('.csv', '.nc')
        elif filename.endswith('.nc'):
            nc_filename = filename
            csv_filename = filename.replace('.nc', '.csv')
        else:
            # 无扩展名，根据format决定
            if format in ['csv', 'both']:
                csv_filename = filename + '.csv'
            if format in ['netcdf', 'both']:
                nc_filename = filename + '.nc'
        
        # 保存CSV格式
        if format in ['csv', 'both'] and csv_filename:
            df.to_csv(csv_filename, index=False)
            logger.info(f"数据已保存到CSV: {csv_filename}")
            success = True
        
        # 保存NetCDF格式
        if format in ['netcdf', 'both'] and nc_filename:
            try:
                from netcdf_handler import NetCDFHandler
                handler = NetCDFHandler()
                
                # 如果format是'both'，先确保CSV已保存
                if format == 'both' and csv_filename and os.path.exists(csv_filename):
                    # 从CSV转换
                    handler.csv_to_netcdf(
                        csv_file=csv_filename,
                        nc_file=nc_filename,
                        lat_points=5,
                        lon_points=5,
                        create_spatial_grid=True
                    )
                else:
                    # 直接从DataFrame创建NetCDF（需要临时CSV）
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
                        temp_csv = tmp.name
                        df.to_csv(temp_csv, index=False)
                    
                    handler.csv_to_netcdf(
                        csv_file=temp_csv,
                        nc_file=nc_filename,
                        lat_points=5,
                        lon_points=5,
                        create_spatial_grid=True
                    )
                    os.remove(temp_csv)  # 清理临时文件
                
                logger.info(f"数据已保存到NetCDF: {nc_filename}")
                success = True
            except ImportError:
                logger.warning("NetCDF处理器未找到，跳过NetCDF保存")
            except Exception as e:
                logger.error(f"保存NetCDF失败: {e}")
        
        return success
    
    def load_data(self, filename='real_yellow_sea_sst_data.csv'):
        """
        从文件加载数据（支持CSV和NetCDF格式）
        
        Args:
            filename: 文件路径（自动检测格式）
        
        Returns:
            pandas DataFrame
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(filename):
                # 尝试查找对应的NetCDF或CSV文件
                if filename.endswith('.csv'):
                    nc_file = filename.replace('.csv', '.nc')
                    if os.path.exists(nc_file):
                        filename = nc_file
                elif filename.endswith('.nc'):
                    csv_file = filename.replace('.nc', '.csv')
                    if os.path.exists(csv_file):
                        filename = csv_file
                else:
                    logger.warning(f"文件 {filename} 不存在")
                    return None
            
            # 根据文件扩展名选择加载方式
            if filename.endswith('.nc'):
                # 加载NetCDF
                try:
                    from netcdf_handler import NetCDFHandler
                    handler = NetCDFHandler()
                    df = pd.read_csv(handler.netcdf_to_csv(filename, filename.replace('.nc', '_temp.csv')))
                    os.remove(filename.replace('.nc', '_temp.csv'))  # 清理临时文件
                    df['date'] = pd.to_datetime(df['date'])
                    logger.info(f"从NetCDF {filename} 加载了 {len(df)} 天的数据")
                    return df
                except ImportError:
                    logger.error("NetCDF处理器未找到，无法加载NetCDF文件")
                    return None
            else:
                # 加载CSV
                df = pd.read_csv(filename)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"从CSV {filename} 加载了 {len(df)} 天的数据")
                return df
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None

def main():
    """主函数"""
    collector = RealDataCollector()
    
    real_data = collector.collect_real_sst_data(
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date']
    )

    if real_data is not None and len(real_data) > 0:
        logger.info("成功收集到真实数据!")
        collector.save_data(real_data, 'real_yellow_sea_sst_data.csv')
        return real_data

    logger.warning("真实数据获取失败，回退到增强模拟数据")
    enhanced_data = collector.create_enhanced_simulated_data(DATA_CONFIG['start_date'])
    collector.save_data(enhanced_data, 'enhanced_yellow_sea_sst_data.csv')
    return enhanced_data

if __name__ == '__main__':
    data = main()
    if data is not None:
        print(f"\n数据收集完成!")
        print(f"数据量: {len(data)} 天")
        print(f"时间范围: {data['date'].min()} 到 {data['date'].max()}")
        print(f"温度范围: {data['mean_sst'].min():.2f}°C 到 {data['mean_sst'].max():.2f}°C")
        print(f"平均温度: {data['mean_sst'].mean():.2f}°C")






