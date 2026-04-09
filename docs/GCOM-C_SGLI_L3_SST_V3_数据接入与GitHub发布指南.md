# GCOM-C/SGLI L3 Sea Surface Temperature (V3) 数据接入与 GitHub 发布指南

本文说明如何用 **GCOM-C/SGLI L3 SST V3** 作为本项目的**唯一训练数据源**（替代增强模拟数据），并在将代码推送到 **GitHub** 时避免泄露大文件与凭证。

---

## 1. 数据产品与版权说明

- **产品**：GCOM-C/SGLI L3 Sea Surface Temperature **Version 3**  
- **说明**：全球海表温度日产品，格点约 **4.6 km**，通常有 **3–4 天**延迟。  
- **Earth Engine 目录**：`JAXA/GCOM-C/L3/OCEAN/SST/V3`  
  - 目录与说明：<https://developers.google.com/earth-engine/datasets/catalog/JAXA_GCOM-C_L3_OCEAN_SST_V3>  
- **原始分发（HDF5）**：JAXA **G-Portal**（需注册）  
  - <https://gportal.jaxa.jp/gpr/>  
- **引用与署名**（发布论文/二次产品时请遵守 JAXA / G-Portal 条款）：  
  - 示例：*Original data for this value added data product was provided by Japan Aerospace Exploration Agency.*  
  - 条款：<https://gportal.jaxa.jp/gpr/index/eula?lang=en>  

---

## 2. 推荐路径：在本机用 Google Earth Engine 生成训练 CSV

本仓库已提供脚本 **`sgli_l3_sst_extract_gee.py`**，在 **黄海矩形范围**（与 `config.py` 中 `YELLOW_SEA_BOUNDS` 一致）内对 `SST_AVE` 做区域统计，输出与训练管线一致的列：

`date`, `mean_sst`, `min_sst`, `max_sst`, `std_sst`, `valid_pixels`

### 2.0 没有银行卡 / 不想绑定 Google Cloud 付款方式？

Google 官方说明：**若以非商业用途注册 Earth Engine 项目，可以不配置结算账号（不要求绑信用卡）**。参考：[Earth Engine access - noncommercial](https://developers.google.com/earth-engine/guides/access)（文中 *“If you register a noncommercial project, no billing configuration is required”*）。

**容易踩坑的地方**：若你从 Cloud 控制台里「随便新建一个项目」，有时会跳出**绑定付款方式**；请尽量走 **Earth Engine 专用入口**，让系统在流程里创建/关联项目并完成**非商业注册问卷**：

1. 打开：<https://console.cloud.google.com/earth-engine>  
2. 按页面引导**创建或选择项目**，并完成 **非商业用途（noncommercial）** 注册 / 问卷（如：<https://console.cloud.google.com/earth-engine/configuration>）。  
3. 确认已启用 **Earth Engine API**（同一流程里通常会带上）。

若你所在地区或账号类型**仍然强制要求绑卡**才能创建任何 Cloud 项目，则**不要死磕 GEE**，改用下面任一方式即可继续毕设（数据仍是真实海温，只是不是 GCOM-C 这一颗卫星）：

| 方式 | 说明 |
|------|------|
| **JAXA G-Portal 下载 HDF5** | 注册 JAXA（一般**不需要**信用卡），手动下载 GCOM-C L3 SST，再自行聚合为 CSV（见本文第 3 节）。 |
| **本项目已有 Open-Meteo** | 运行 `py data_refresh.py refresh`，得到 `real_yellow_sea_sst_data.csv`（无需 GEE）。训练时若不存在 `sgli_*.csv`，仍会使用 `data/yellow_sea_sst_data.csv`；可把 Open-Meteo 导出的 CSV 复制/命名为训练用文件。 |

### 2.1 前置条件

1. Google 账号，并开通 **Earth Engine**：  
   <https://console.cloud.google.com/earth-engine>  
2. 记录你的 **Google Cloud 项目 ID**（Project ID）。  
3. 本机安装依赖（建议在虚拟环境中）：

   ```bash
   pip install -r requirements.txt
   ```

4. 命令行完成 Earth Engine 登录：

   ```bash
   earthengine authenticate
   ```

### 2.2 设置项目 ID（Windows PowerShell 示例）

```powershell
$env:EE_PROJECT = "你的-google-cloud-项目-id"
```

也可在运行脚本时用参数 `--project` 传入（见下）。

### 2.3 运行提取

在项目根目录执行：

```powershell
py sgli_l3_sst_extract_gee.py --start 2018-01-22 --end 2025-12-31
```

默认输出：**`data/sgli_yellow_sea_sst_daily.csv`**。

常用参数：

| 参数 | 含义 |
|------|------|
| `--start` / `--end` | 起止日期（产品自 **2018-01-22** 起） |
| `--out` | 自定义输出路径 |
| `--project` | Earth Engine 项目 ID |
| `--include-night` | 默认只用白天轨（`SATELLITE_DIRECTION=D`）；加此参数则不过滤 |
| `--chunk-days` | 分段天数，避免单次请求过大（默认 45） |

### 2.4 SST 物理量换算

脚本中 `SST_AVE` 按 Earth Engine 官方示例转换为摄氏度：

`SST_C = SST_AVE * 0.0012 - 10`

并对 `SST_QA_flag` 的 **Bits 0–1** 做掩膜：优先保留 **纯水 / 大部分为水** 像素，弱化陆地与强海岸混合像元。

### 2.5 与训练 / Web 的衔接

- **`training.py`** 会通过 `config.resolve_training_data_csv()` **优先**使用  
  `data/sgli_yellow_sea_sst_daily.csv`（若存在）。  
- **`web_app_real_data.py`** 同样 **优先**加载该文件。  

生成 CSV 后，直接重新训练即可：

```powershell
py training.py
```

启动 Web：

```powershell
py web_app_real_data.py
```

---

## 3. 备选路径：从 JAXA G-Portal 下载 HDF5 自行聚合

适用于不能使用 Earth Engine、但必须使用 **官方原始 L3 文件** 的场景。

### 3.1 流程概要

1. 在 G-Portal 注册并登录，检索 **GCOM-C SGLI L3 SST** 相关产品（V3）。  
2. 按日或按时间段下载 **HDF5**（具体文件名与内部数据集路径以产品说明为准）。  
3. 自行编写或使用脚本：  
   - 读取 HDF5 中 SST 与 QA 波段；  
   - 截取 **119°E–127°E，32°N–41°N**；  
   - 按日计算区域 **mean / min / max / std / 有效像元数**；  
   - 写出与本项目相同列名的 CSV，保存为  
     **`data/sgli_yellow_sea_sst_daily.csv`**（或复制为 `data/yellow_sea_sst_data.csv` 并改 `resolve_training_data_csv` 逻辑）。  

> **说明**：HDF5 内部变量名因产品版本与处理链可能不同，需对照 JAXA 产品手册；本仓库未绑定某一固定文件名，故未内置通用 HDF5 解析脚本。若你提供一份样本 HDF5 路径与 `h5dump -n` 结构，可再扩展专用转换工具。

常用依赖：`h5py`、`numpy`（可选 `xarray` + `netcdf4`/`h5netcdf` 视文件结构而定）。

---

## 4. 完全停用模拟数据时的注意点

- **`data_refresh.py`** 在 Open-Meteo 失败时会 **回退到增强模拟数据**。若你希望 **绝不**再生成模拟数据，请：  
  - 仅用 **`sgli_l3_sst_extract_gee.py`** 更新 `data/sgli_yellow_sea_sst_daily.csv`；  
  - 或修改 `data_refresh.py`，去掉 `create_enhanced_simulated_data` 分支（需自行改代码）。  
- **`web_app_real_data.py`** 在找不到任何 CSV 时仍会 **现场生成增强模拟数据**；生产环境建议 **保证** `data/sgli_yellow_sea_sst_daily.csv` 已存在，或同样改掉该回退逻辑。

---

## 5. 常见问题

**Q：提取很慢或超时？**  
减小 `--chunk-days`（例如 30），或缩短 `--start`/`--end` 区间分段运行后再用 pandas 合并 CSV。

**Q：某些日期没有行？**  
云覆盖、全掩膜、或产品缺轨会导致当日统计为空；可在预处理阶段对缺失日期做插值（本项目 `data_preprocessing.py` 已含 KNN 等逻辑）。

**Q：与 Open-Meteo 数据能否混用？**  
可以，但不建议无文档地混用。选定 **单一物理数据源** 作为论文/毕设主线更清晰。

---

## 6. 一键命令小结（Earth Engine 路径）

```powershell
$env:EE_PROJECT = "你的项目ID"
earthengine authenticate
py sgli_l3_sst_extract_gee.py --start 2018-01-22 --end 2025-12-31
py training.py
py web_app_real_data.py
```

完成以上步骤后，即可在本地用 **GCOM-C/SGLI L3 SST V3** 替代原模拟数据，并将**不含大文件与密钥**的代码仓安全推送到 GitHub。
