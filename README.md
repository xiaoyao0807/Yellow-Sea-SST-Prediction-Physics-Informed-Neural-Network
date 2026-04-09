# Yellow Sea SST Prediction — Physics-Informed Neural Network

基于 **物理信息神经网络（PINN）** 和 **JAXA GCOM-C/SGLI L3 卫星数据** 的黄海海表温度预测系统。

支持 **1–90 天** 海温预测，提供响应式 Web 可视化界面，测试集 **R² = 0.978**，RMSE = 1.29 °C。

---

## Preview

```
首页        →  实时温度仪表盘 + 系统概览
预测页      →  自定义 1–90 天，交互式曲线 + 明细表
分析页      →  历史趋势、月度分布、统计面板
关于页      →  模型信息、物理约束说明、数据来源
```

---

## Highlights

- **真实卫星数据**：GCOM-C/SGLI L3 SST V3（JAXA），通过 Google Earth Engine 提取，覆盖 2018-01 ~ 2026-04
- **物理约束驱动**：4 类约束（温度范围、季节周期、时间平滑、热惯性）+ warm-up 训练策略
- **端到端管线**：数据采集 → 特征工程 → PINN 训练 → 评估 → Web 服务，一键复现
- **响应式前端**：Flask + Chart.js，中英双语，桌面/平板/手机自适应

---

## Quick Start

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 获取卫星数据（首次）

在 [Google Earth Engine Code Editor](https://code.earthengine.google.com/) 中运行导出脚本（详见 `docs/` 下的数据接入指南），将 CSV 放入 `data/sgli_yellow_sea_sst_daily.csv`。

### 3. 数据清洗

```bash
python prepare_sgli_data.py
```

### 4. 训练模型

```bash
python training.py
```

### 5. 启动 Web 服务

```bash
python web_app_real_data.py
```

浏览器打开 **http://127.0.0.1:5000** 即可查看。

---

## Reproducibility Checklist

为确保你本地结果和仓库说明一致，建议按下面顺序执行并核对：

1. `data/sgli_yellow_sea_sst_daily.csv` 存在，且日期覆盖到最近可用观测日  
2. `python prepare_sgli_data.py` 执行成功（完成低质量过滤 + 缺失插值）  
3. `python training.py` 输出 `models/best_model.pth` 和 `results/model_evaluation.csv`  
4. `python web_app_real_data.py` 启动成功并显示 `Running on http://127.0.0.1:5000`  
5. 浏览器可查看首页、预测页和分析页图表

---

## API Endpoints

后端基于 Flask 提供 JSON API，便于前端和二次集成调用：

| Endpoint | Method | Description | Example |
|---|---|---|---|
| `/api/health` | GET | 服务健康状态、数据源、时间范围 | `/api/health` |
| `/api/data/historical` | GET | 历史数据（可选 `days`） | `/api/data/historical?days=180` |
| `/api/data/statistics` | GET | 统计指标（均值、极值、季节分布） | `/api/data/statistics` |
| `/api/predict` | GET | 未来预测（`days=1~90`） | `/api/predict?days=30` |
| `/api/model_info` | GET | 模型与约束说明、数据来源 | `/api/model_info` |

---

## Common Workflows

### A) 只更新数据，不改代码

1. 在 GEE 导出新 CSV  
2. 覆盖 `data/sgli_yellow_sea_sst_daily.csv`  
3. 运行 `python prepare_sgli_data.py`  
4. 运行 `python training.py`  

### B) 仅演示 Web（不重新训练）

1. 确保已有可用数据 CSV  
2. 直接运行 `python web_app_real_data.py`  
3. 浏览器打开 `http://127.0.0.1:5000`

### C) 调整物理约束

1. 在 `config.py` 修改 `PHYSICS_CONFIG`（`lambda_*`、`warmup_epochs` 等）  
2. 运行 `python training.py`  
3. 比较 `results/model_evaluation.csv` 的指标变化

---

## Troubleshooting

| 问题 | 常见原因 | 解决方案 |
|---|---|---|
| 训练报 `data file not found` | 数据 CSV 不在 `data/` 下 | 检查 `data/sgli_yellow_sea_sst_daily.csv` 路径 |
| 训练指标突然变差 | 新数据质量低或时间跨度太短 | 重新导出数据，确认 `valid_pixels` 过滤后样本量 |
| Web 启动但图表为空 | API 请求失败或前端缓存 | 刷新页面、检查控制台、访问 `/api/health` |
| GEE 导出失败 | 任务未 Run 或配额限制 | 在 GEE Tasks 面板重新 Run，缩短导出时间范围 |
| 端口 5000 被占用 | 本地已有 Flask 进程 | 关闭旧进程或修改端口后再启动 |

---

## Project Structure

```
├── config.py                    # 全局配置（模型、物理约束、数据路径）
├── sgli_l3_sst_extract_gee.py   # Earth Engine 数据提取脚本
├── prepare_sgli_data.py         # SGLI 原始 CSV 清洗（过滤+插值）
├── real_data_collector.py       # Open-Meteo 备选数据采集
├── data_refresh.py              # 数据刷新入口
├── data_preprocessing.py        # 特征工程（时间/滞后/物理特征）
├── pinn_model.py                # PINN 模型定义 + 物理约束 + 训练器
├── training.py                  # 训练流水线（预处理→训练→评估→保存）
├── evaluation.py                # 评估指标计算
├── web_app_real_data.py         # Flask Web 服务
├── requirements.txt             # Python 依赖
│
├── data/                        # 训练数据（.gitignore，本地生成）
│   └── sgli_yellow_sea_sst_daily.csv
├── models/                      # 模型权重与元数据
├── results/                     # 评估与预测 CSV
├── plots/                       # 可视化图表
├── templates/                   # HTML 模板（中文 + 英文）
├── static/                      # 前端 CSS / JS
└── docs/                        # 数据接入与 GitHub 发布指南
```

---

## Data Source

**GCOM-C/SGLI L3 Sea Surface Temperature (V3)**

- 产品方：JAXA（日本宇宙航空研究开发机构）
- 分辨率：~4.6 km，日频
- 覆盖范围：全球海洋，2018-01-22 至今
- Earth Engine 目录：`JAXA/GCOM-C/L3/OCEAN/SST/V3`
- 本项目区域：黄海（119°E–127°E，32°N–41°N）

> *Original data provided by Japan Aerospace Exploration Agency.*
> 详细获取步骤见 [`docs/GCOM-C_SGLI_L3_SST_V3_数据接入与GitHub发布指南.md`](docs/GCOM-C_SGLI_L3_SST_V3_数据接入与GitHub发布指南.md)

---

## PINN Model

### Architecture

```
Input (25 features) → [128] → Tanh → [128] → Tanh → [64] → Tanh → [32] → Tanh → [1] (SST)
```

### Physics Constraints

| 约束 | 公式 | 物理含义 |
|------|------|---------|
| 温度范围 | penalty for pred ∉ [−3, 31] °C | 黄海 SST 的物理合理区间 |
| 季节周期 | corr(pred, cos(doy)) < −0.3 | 冬冷夏热的年周期规律 |
| 时间平滑 | ‖pred − 2·lag₁ + lag₂‖² | 温度变化"加速度"应小 |
| 热惯性 | \|pred − lag₁\| ≤ 2°C/day | 海洋混合层热容大，日变化有限 |

### Training Strategy

- **Optimizer**: Adam (lr = 0.001)
- **Scheduler**: ReduceLROnPlateau (factor 0.5, patience 20)
- **Early Stopping**: patience 50
- **Physics Warm-up**: 前 30 轮线性增加物理损失权重
- **Gradient Clipping**: max_norm = 1.0

---

## Performance

| Metric | Value |
|--------|-------|
| R² | **0.9778** |
| RMSE | **1.29 °C** |
| MAE | **1.00 °C** |
| MAPE | 20.05% |
| 训练数据 | 2995 天 (2018-01 ~ 2026-04) |
| 预测范围 | 1–90 天 |

---

## Web Interface

| 模块 | 功能 |
|------|------|
| 首页 | 系统介绍、实时温度仪表盘 |
| 预测 | 自定义 1–90 天预测，曲线图 + 明细表 |
| 分析 | 历史趋势、月度季节性柱状图、统计面板 |
| 关于 | 模型信息、物理约束说明、数据来源 |

---

## License & Citation

数据使用请遵守 [JAXA G-Portal 使用条款](https://gportal.jaxa.jp/gpr/index/eula?lang=en)。

引用卫星数据：
> Kurihara, Y. (2020). GCOM-C/SGLI Sea Surface Temperature (SST) ATBD (Version 2).

---

## Acknowledgments

- **JAXA** — GCOM-C/SGLI L3 SST V3 数据产品
- **Google Earth Engine** — 卫星数据提取平台
- **PyTorch** — 深度学习框架

---

## Roadmap

- [ ] 支持多变量输入（风速、盐度、海表高度异常）  
- [ ] 增加按季节/月份分组评估报告  
- [ ] 提供 Docker 化部署与一键启动脚本  
- [ ] 增加模型对比基线（LSTM/Transformer/LightGBM）  
- [ ] 增加置信区间（Prediction Interval）而非单点置信度
