"""
清洗 GCOM-C/SGLI L3 SST V3 导出的原始 CSV，使其适合训练管线。

处理内容：
1. 过滤有效像元数过少（<10）的不可靠记录
2. 将日期补全为连续序列（卫星因云覆盖缺失的日期）
3. 对缺失日期做线性插值
4. 保存为训练管线可直接使用的 CSV
"""
import os
import pandas as pd
import numpy as np


def prepare(
    raw_csv: str = "data/sgli_yellow_sea_sst_daily.csv",
    out_csv: str = "data/sgli_yellow_sea_sst_daily.csv",
    min_valid_pixels: int = 10,
):
    df = pd.read_csv(raw_csv)
    df["date"] = pd.to_datetime(df["date"])
    total_raw = len(df)

    df = df[df["valid_pixels"] >= min_valid_pixels].copy()
    after_filter = len(df)
    print(f"过滤有效像元 < {min_valid_pixels} 的记录: {total_raw} -> {after_filter} (去掉 {total_raw - after_filter} 行)")

    date_range = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    df = df.set_index("date").reindex(date_range)
    df.index.name = "date"

    missing_days = df["mean_sst"].isna().sum()
    print(f"补全日期后共 {len(df)} 天，其中 {missing_days} 天需要插值 ({missing_days/len(df)*100:.1f}%)")

    for col in ["mean_sst", "min_sst", "max_sst", "std_sst"]:
        df[col] = df[col].interpolate(method="linear", limit_direction="both")

    df["valid_pixels"] = df["valid_pixels"].fillna(0).astype(int)

    df = df.reset_index()
    df = df.rename(columns={"index": "date"})

    df.to_csv(out_csv, index=False)
    print(f"\n已保存: {out_csv}")
    print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"总天数: {len(df)}")
    print(f"温度范围: {df['mean_sst'].min():.2f}°C ~ {df['mean_sst'].max():.2f}°C")
    print(f"平均温度: {df['mean_sst'].mean():.2f}°C")


if __name__ == "__main__":
    prepare()
