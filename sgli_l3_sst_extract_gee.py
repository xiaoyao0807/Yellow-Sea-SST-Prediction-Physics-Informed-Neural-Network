"""
从 Google Earth Engine 提取 GCOM-C/SGLI L3 Sea Surface Temperature (V3) 日值，
在黄海研究区内聚合为与训练管线一致的 CSV（mean/min/max/std/valid_pixels）。

数据源与官方示例一致：JAXA/GCOM-C/L3/OCEAN/SST/V3
SST 物理量按 Earth Engine 文档：SST_AVE * 0.0012 - 10 （单位 °C）

使用前请完成 Earth Engine 注册，并设置环境变量：
  set EE_PROJECT=你的_Google_Cloud_项目_ID
然后运行：earthengine authenticate

依赖：earthengine-api（已在 requirements.txt）
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

from config import DATA_CONFIG, YELLOW_SEA_BOUNDS, OUTPUT_CONFIG


def _build_geometry():
    b = YELLOW_SEA_BOUNDS
    return {
        "west": b["west"],
        "south": b["south"],
        "east": b["east"],
        "north": b["north"],
    }


def _date_chunks(start: str, end: str, chunk_days: int):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    cur = s
    while cur <= e:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), e)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)


def extract_to_dataframe(
    start_date: str,
    end_date: str,
    *,
    daytime_only: bool = True,
    chunk_days: int = 45,
    ee_project: str | None = None,
) -> pd.DataFrame:
    import ee

    project = ee_project or os.environ.get("EE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise RuntimeError(
            "未设置 Earth Engine 项目 ID。请设置环境变量 EE_PROJECT（或 GOOGLE_CLOUD_PROJECT），"
            "或在命令行传入 --project。"
        )

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        raise RuntimeError(
            f"Earth Engine 初始化失败: {exc}\n"
            "请先运行: earthengine authenticate\n"
            "并确认已在 https://console.cloud.google.com/earth-engine 开通 Earth Engine。"
        ) from exc

    bounds = _build_geometry()
    geom = ee.Geometry.Rectangle(
        [bounds["west"], bounds["south"], bounds["east"], bounds["north"]]
    )

    def image_to_feature(img: ee.Image) -> ee.Feature:
        date_str = img.date().format("YYYY-MM-dd")
        # 与官方 Code Editor 示例一致的线性变换
        sst = img.select("SST_AVE").multiply(0.0012).add(-10)
        qa = img.select("SST_QA_flag")
        # Bits 0-1: 0=纯水, 1=大部分为水 —— 用于弱化陆地/海岸像素
        mask = qa.bitwiseAnd(3).lte(1)
        sst_masked = sst.updateMask(mask)

        stats = sst_masked.reduceRegion(
            reducer=ee.Reducer.mean()
            .combine(reducer2=ee.Reducer.minMax(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True)
            .combine(reducer2=ee.Reducer.count(), sharedInputs=True),
            geometry=geom,
            scale=4638.3,
            maxPixels=1e13,
            bestEffort=True,
        )
        return ee.Feature(None, {"date": date_str}).set(stats)

    rows = []
    for c_start, c_end in _date_chunks(start_date, end_date, chunk_days):
        col = ee.ImageCollection("JAXA/GCOM-C/L3/OCEAN/SST/V3").filterDate(
            c_start, _next_day(c_end)
        )
        col = col.filterBounds(geom)
        if daytime_only:
            col = col.filter(ee.Filter.eq("SATELLITE_DIRECTION", "D"))

        fc = ee.FeatureCollection(col.map(image_to_feature))
        info = fc.getInfo()
        for f in info.get("features", []):
            p = f.get("properties") or {}
            row = _properties_to_row(p)
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["date", "mean_sst", "min_sst", "max_sst", "std_sst", "valid_pixels"]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset="date", keep="first")
    df = df.reset_index(drop=True)
    return df


def _next_day(date_str: str) -> str:
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def _properties_to_row(p: dict) -> dict | None:
    date_val = p.get("date")
    if not date_val:
        return None

    mean_k = "SST_AVE_mean"
    min_k = "SST_AVE_min"
    max_k = "SST_AVE_max"
    std_k = "SST_AVE_stdDev"
    cnt_k = "SST_AVE_count"

    mean_v = p.get(mean_k)
    if mean_v is None:
        return None

    def num(x):
        if x is None:
            return float("nan")
        try:
            return float(x)
        except (TypeError, ValueError):
            return float("nan")

    return {
        "date": date_val,
        "mean_sst": num(mean_v),
        "min_sst": num(p.get(min_k)),
        "max_sst": num(p.get(max_k)),
        "std_sst": num(p.get(std_k)),
        "valid_pixels": int(p.get(cnt_k) or 0),
    }


def main():
    parser = argparse.ArgumentParser(description="GCOM-C SGLI L3 SST V3 → 黄海日值 CSV（Earth Engine）")
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD（默认 config.DATA_CONFIG）")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认 config.DATA_CONFIG）")
    parser.add_argument("--out", default=None, help="输出 CSV 路径（默认 data/sgli_yellow_sea_sst_daily.csv）")
    parser.add_argument("--project", default=None, help="Google Cloud / Earth Engine 项目 ID")
    parser.add_argument(
        "--include-night",
        action="store_true",
        help="不过滤卫星方向（默认仅白天 SATELLITE_DIRECTION=D）",
    )
    parser.add_argument("--chunk-days", type=int, default=45, help="每次拉取的跨度（天），过大可能超时")
    args = parser.parse_args()

    start = args.start or DATA_CONFIG["start_date"]
    end = args.end or DATA_CONFIG["end_date"]
    out = args.out or os.path.join(OUTPUT_CONFIG["data_save_path"], "sgli_yellow_sea_sst_daily.csv")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    print(f"Earth Engine 提取 GCOM-C L3 SST V3: {start} ~ {end}")
    print(f"区域: {YELLOW_SEA_BOUNDS}")
    df = extract_to_dataframe(
        start,
        end,
        daytime_only=not args.include_night,
        chunk_days=args.chunk_days,
        ee_project=args.project,
    )

    if df.empty:
        print("未得到任何记录。请检查日期范围（产品自 2018-01-22 起）、区域与网络。")
        sys.exit(1)

    df.to_csv(out, index=False)
    print(f"已写入 {len(df)} 行: {out}")
    print(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")


if __name__ == "__main__":
    main()
