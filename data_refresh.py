"""
数据刷新脚本
用于更新和刷新黄海海水温度数据
"""
import os
import sys

import pandas as pd

from config import DATA_CONFIG
from real_data_collector import RealDataCollector


def refresh_data():
    """刷新数据"""
    print("🔄 开始刷新黄海海水温度数据...")

    collector = RealDataCollector()

    print("📡 正在通过 Open-Meteo Marine 获取真实SST数据...")
    real_data = collector.collect_real_sst_data(
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date']
    )

    if real_data is not None and len(real_data) > 0:
        collector.save_data(real_data, 'real_yellow_sea_sst_data.csv')
        print(f"✅ 成功获取并保存真实数据: {len(real_data)} 天")
        print(f"📅 时间范围: {real_data['date'].min().strftime('%Y-%m-%d')} 至 {real_data['date'].max().strftime('%Y-%m-%d')}")
        print("💾 文件: real_yellow_sea_sst_data.csv")
        return real_data

    print("⚠️ 真实数据获取失败，改用增强模拟数据")
    enhanced_data = collector.create_enhanced_simulated_data(DATA_CONFIG['start_date'], DATA_CONFIG['end_date'])
    collector.save_data(enhanced_data, 'enhanced_yellow_sea_sst_data.csv')

    print(f"✅ 增强模拟数据创建完成: {len(enhanced_data)} 天")
    print("💾 文件: enhanced_yellow_sea_sst_data.csv")
    return enhanced_data


def show_data_info():
    """显示数据信息"""
    print("\n📊 当前数据文件状态:")

    files_to_check = [
        'real_yellow_sea_sst_data.csv',
        'enhanced_yellow_sea_sst_data.csv'
    ]

    for filename in files_to_check:
        if os.path.exists(filename):
            try:
                data = pd.read_csv(filename)
                data['date'] = pd.to_datetime(data['date'])

                print(f"✅ {filename}:")
                print(f"   📅 时间范围: {data['date'].min().strftime('%Y-%m-%d')} 到 {data['date'].max().strftime('%Y-%m-%d')}")
                print(f"   📈 数据量: {len(data)} 天")
                print(f"   🌡️ 温度范围: {data['mean_sst'].min():.2f}°C 到 {data['mean_sst'].max():.2f}°C")
                print(f"   📊 平均温度: {data['mean_sst'].mean():.2f}°C")
                print(f"   📝 文件大小: {os.path.getsize(filename) / 1024:.1f} KB")
                print()
            except Exception as exc:
                print(f"❌ {filename}: 读取失败 - {exc}")
        else:
            print(f"❌ {filename}: 文件不存在")


def main():
    """主函数"""
    print("🌊 黄海海水温度数据管理系统")
    print("=" * 50)

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == 'refresh':
            data = refresh_data()
            if data is not None and len(data) > 0:
                print(f"\n🎉 数据刷新完成!")
                print(f"📊 总数据量: {len(data)} 天")
                print(f"📅 最新数据: {data['date'].max().strftime('%Y-%m-%d')}")

        elif command == 'info':
            show_data_info()

        elif command == 'help':
            print("\n📖 使用说明:")
            print("  py data_refresh.py refresh  - 刷新数据")
            print("  py data_refresh.py info     - 显示数据信息")
            print("  py data_refresh.py help     - 显示帮助")
            print("\nGCOM-C/SGLI L3 SST V3（推荐替代模拟数据）:")
            print("  见 docs/GCOM-C_SGLI_L3_SST_V3_数据接入与GitHub发布指南.md")
            print("  py sgli_l3_sst_extract_gee.py --start 2018-01-22 --end 2025-12-31")

        else:
            print(f"❌ 未知命令: {command}")
            print("使用 'py data_refresh.py help' 查看帮助")

    else:
        # 默认显示数据信息
        show_data_info()
        print("\n💡 提示:")
        print("  使用 'py data_refresh.py refresh' 刷新数据")
        print("  使用 'py data_refresh.py info' 查看详细信息")


if __name__ == '__main__':
    main()
