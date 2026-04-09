// 黄海海水温度预测系统 - JavaScript文件

const TEXT_MAP = {
    zh: {
        months: ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月'],
        notifications: {
            predictionSuccess: '预测完成！',
            predictionFailure: '预测失败',
            predictionErrorPrefix: '预测失败: ',
            analysisError: '分析数据加载失败'
        },
        charts: {
            predictionTitle: '海水温度预测趋势',
            historicalTitle: '历史温度趋势',
            seasonalTitle: '月度温度分布',
            yAxisTemp: '温度 (°C)',
            xAxisDate: '日期',
            xAxisMonth: '月份',
            predictionDataset: '预测温度',
            historicalMean: '平均温度',
            historicalMax: '最高温度',
            historicalMin: '最低温度'
        },
        stats: {
            totalDays: '样本天数',
            avgTemp: '平均温度',
            minTemp: '最低温度',
            maxTemp: '最高温度',
            stdTemp: '温度标准差',
            startDate: '数据开始日期',
            endDate: '数据结束日期',
            currentTemp: '当前最新温度'
        },
        metrics: {
            r2: 'R² 指标',
            rmse: 'RMSE',
            mae: 'MAE',
            mape: 'MAPE',
            tempViolation: '温度范围违反率',
            seasonalConsistency: '季节一致性',
            smoothness: '平滑性评分',
            conservation: '守恒符合度',
            dataSource: '数据来源'
        },
        temperatureStatus: {
            cold: '寒冷',
            cool: '凉爽',
            warm: '温暖',
            hot: '炎热'
        }
    },
    en: {
        months: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        notifications: {
            predictionSuccess: 'Prediction completed!',
            predictionFailure: 'Prediction failed',
            predictionErrorPrefix: 'Prediction failed: ',
            analysisError: 'Failed to load analysis data'
        },
        charts: {
            predictionTitle: 'Sea Surface Temperature Forecast',
            historicalTitle: 'Historical Temperature Trends',
            seasonalTitle: 'Monthly Temperature Distribution',
            yAxisTemp: 'Temperature (°C)',
            xAxisDate: 'Date',
            xAxisMonth: 'Month',
            predictionDataset: 'Predicted Temperature',
            historicalMean: 'Mean Temperature',
            historicalMax: 'Maximum Temperature',
            historicalMin: 'Minimum Temperature'
        },
        stats: {
            totalDays: 'Total Days',
            avgTemp: 'Average Temperature',
            minTemp: 'Minimum Temperature',
            maxTemp: 'Maximum Temperature',
            stdTemp: 'Temperature Std Dev',
            startDate: 'Start Date',
            endDate: 'End Date',
            currentTemp: 'Latest Temperature'
        },
        metrics: {
            r2: 'R² Score',
            rmse: 'RMSE',
            mae: 'MAE',
            mape: 'MAPE',
            tempViolation: 'Temperature Range Violation',
            seasonalConsistency: 'Seasonal Consistency',
            smoothness: 'Smoothness Score',
            conservation: 'Conservation Score',
            dataSource: 'Data Source'
        },
        temperatureStatus: {
            cold: 'Cold',
            cool: 'Cool',
            warm: 'Warm',
            hot: 'Hot'
        }
    }
};

class TemperaturePredictionApp {
    constructor() {
        this.currentSection = 'home';
        this.charts = {};
        this.currentData = null;
        const htmlLang = (document.documentElement.lang || '').toLowerCase();
        this.lang = htmlLang.startsWith('en') ? 'en' : 'zh';
        this.text = TEXT_MAP[this.lang] || TEXT_MAP.zh;
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.updateCurrentTemperature();
    }

    setupEventListeners() {
        // 导航菜单
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = e.target.getAttribute('href').substring(1);
                this.showSection(section);
            });
        });

        // 移动端菜单切换
        const navToggle = document.querySelector('.nav-toggle');
        if (navToggle) {
            navToggle.addEventListener('click', () => {
                const navMenu = document.querySelector('.nav-menu');
                navMenu.classList.toggle('active');
            });
        }

        // 预测按钮
        const predictBtn = document.querySelector('.btn-primary');
        if (predictBtn) {
            predictBtn.addEventListener('click', () => {
                this.makePrediction();
            });
        }
    }

    showSection(sectionId) {
        // 隐藏所有部分
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });

        // 显示目标部分
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
            this.currentSection = sectionId;

            // 更新导航状态
            document.querySelectorAll('.nav-link').forEach(link => {
                link.classList.remove('active');
            });
            document.querySelector(`[href="#${sectionId}"]`).classList.add('active');

            // 加载部分特定数据
            this.loadSectionData(sectionId);
        }
    }

    async loadInitialData() {
        try {
            // 加载统计信息
            const statsResponse = await fetch('/api/data/statistics');
            const statsData = await statsResponse.json();
            
            if (statsData.success) {
                this.currentData = statsData;
                this.updateCurrentTemperature();
            }
        } catch (error) {
            console.error('加载初始数据失败:', error);
        }
    }

    async loadSectionData(sectionId) {
        switch (sectionId) {
            case 'analysis':
                await this.loadAnalysisData();
                break;
            case 'about':
                await this.loadAboutData();
                break;
        }
    }

    async loadAnalysisData() {
        try {
            this.showLoading(true);

            // 并行加载多个数据源
            const [
                historicalResponse,
                seasonalResponse,
                statsResponse,
                performanceResponse
            ] = await Promise.all([
                fetch('/api/data/historical?days=365'),
                fetch('/api/data/seasonal'),
                fetch('/api/data/statistics'),
                fetch('/api/model/performance')
            ]);

            const [
                historicalData,
                seasonalData,
                statsData,
                performanceData
            ] = await Promise.all([
                historicalResponse.json(),
                seasonalResponse.json(),
                statsResponse.json(),
                performanceResponse.json()
            ]);

            if (historicalData.success) {
                this.createHistoricalChart(historicalData.data);
            }

            if (seasonalData.success) {
                this.createSeasonalChart(seasonalData.seasonal_data);
            }

            if (statsData.success) {
                this.updateStatistics(statsData);
            }

            if (performanceData.success) {
                this.updatePerformanceMetrics(performanceData);
            }

        } catch (error) {
            console.error('Failed to load analysis data:', error);
            this.showNotification(this.text.notifications.analysisError, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async loadAboutData() {
        try {
            const response = await fetch('/api/model/info');
            const data = await response.json();
            
            if (data.success) {
                // 可以在这里更新关于页面的模型信息
                console.log('模型信息:', data.model_info);
            }
        } catch (error) {
            console.error('加载模型信息失败:', error);
        }
    }

    updateCurrentTemperature() {
        if (this.currentData && this.currentData.temperature_stats) {
            const currentTemp = this.currentData.current_temperature ?? this.currentData.temperature_stats.mean;
            const tempElement = document.getElementById('currentTemp');
            const gaugeElement = document.getElementById('currentTempGauge');
            
            if (tempElement) {
                tempElement.textContent = currentTemp.toFixed(1);
            }
            
            if (gaugeElement) {
                // 更新温度计显示 (0-30°C映射到0-100%)
                const percentage = (currentTemp / 30) * 100;
                gaugeElement.style.background = `conic-gradient(from 0deg, #e74c3c 0%, #f39c12 25%, #f1c40f 50%, #2ecc71 75%, #3498db ${percentage}%)`;
            }
        }
    }

    async makePrediction() {
        try {
            this.showLoading(true);

            const daysAhead = parseInt(document.getElementById('predictionDays').value);
            
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    days_ahead: daysAhead
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayPredictionResults(data);
                this.showNotification(this.text.notifications.predictionSuccess, 'success');
            } else {
                this.showNotification(data.error || this.text.notifications.predictionFailure, 'error');
            }

        } catch (error) {
            console.error('预测失败:', error);
            this.showNotification(this.text.notifications.predictionErrorPrefix + error.message, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    displayPredictionResults(data) {
        const resultsContainer = document.getElementById('predictionResults');
        const modelTypeElement = document.getElementById('modelType');
        const tableBody = document.getElementById('predictionTableBody');

        // 显示结果容器
        resultsContainer.style.display = 'block';

        // 更新模型类型
        if (modelTypeElement) {
            modelTypeElement.textContent = data.prediction_info.model_type;
        }

        // 创建预测图表
        this.createPredictionChart(data.predictions);

        // 更新预测表格
        if (tableBody) {
            tableBody.innerHTML = '';
            data.predictions.forEach(prediction => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${prediction.date}</td>
                    <td>${prediction.temperature}°C</td>
                    <td>${(prediction.confidence * 100).toFixed(1)}%</td>
                    <td>
                        <span class="status-badge ${this.getTemperatureStatus(prediction.temperature)}">
                            ${this.getTemperatureStatusText(prediction.temperature)}
                        </span>
                    </td>
                `;
                tableBody.appendChild(row);
            });
        }

        // 滚动到结果
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    createPredictionChart(predictions) {
        const ctx = document.getElementById('predictionChart');
        if (!ctx) return;

        // 销毁现有图表
        if (this.charts.prediction) {
            this.charts.prediction.destroy();
        }

        const labels = predictions.map(p => p.date);
        const temperatures = predictions.map(p => p.temperature);
        const confidences = predictions.map(p => p.confidence);

        this.charts.prediction = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: this.text.charts.predictionDataset,
                    data: temperatures,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#3498db',
                    pointBorderColor: '#2980b9',
                    pointRadius: 5,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: this.text.charts.predictionTitle,
                        font: {
                            size: 16,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                        text: this.text.charts.yAxisTemp
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: this.text.charts.xAxisDate
                        },
                        grid: {
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    createHistoricalChart(data) {
        const ctx = document.getElementById('historicalChart');
        if (!ctx) return;

        // 销毁现有图表
        if (this.charts.historical) {
            this.charts.historical.destroy();
        }

        const records = Array.isArray(data) ? data : [];
        const sortedRecords = records
            .map(item => ({
                date: item.date,
                mean: item.mean_sst ?? item.mean,
                min: item.min_sst ?? item.min,
                max: item.max_sst ?? item.max
            }))
            .filter(item => item.date)
            .sort((a, b) => new Date(a.date) - new Date(b.date));

        const recentRecords = sortedRecords.slice(-100);
        const labels = recentRecords.map(item => item.date);
        const temperatures = recentRecords.map(item => item.mean);
        const minTemps = recentRecords.map(item => item.min);
        const maxTemps = recentRecords.map(item => item.max);

        this.charts.historical = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: this.text.charts.historicalMean,
                    data: temperatures,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.1
                }, {
                    label: this.text.charts.historicalMax,
                    data: maxTemps,
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.1
                }, {
                    label: this.text.charts.historicalMin,
                    data: minTemps,
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 1,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: this.text.charts.historicalTitle,
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: this.text.charts.yAxisTemp
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: this.text.charts.xAxisDate
                        }
                    }
                }
            }
        });
    }

    createSeasonalChart(seasonalData) {
        const ctx = document.getElementById('seasonalChart');
        if (!ctx) return;

        // 销毁现有图表
        if (this.charts.seasonal) {
            this.charts.seasonal.destroy();
        }

        const sortedSeasonal = Array.isArray(seasonalData)
            ? seasonalData.slice().sort((a, b) => a.month - b.month)
            : [];

        const meanTemps = new Array(12).fill(null);
        const minTemps = new Array(12).fill(null);
        const maxTemps = new Array(12).fill(null);

        sortedSeasonal.forEach(item => {
            const monthIndex = item.month - 1;
            if (monthIndex >= 0 && monthIndex < 12) {
                meanTemps[monthIndex] = item.mean_temp;
                minTemps[monthIndex] = item.min_temp;
                maxTemps[monthIndex] = item.max_temp;
            }
        });

        this.charts.seasonal = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.text.months,
                datasets: [{
                    label: this.text.charts.historicalMean,
                    data: meanTemps,
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    borderColor: '#3498db',
                    borderWidth: 1,
                    spanGaps: true
                }, {
                    label: this.text.charts.historicalMax,
                    data: maxTemps,
                    backgroundColor: 'rgba(231, 76, 60, 0.8)',
                    borderColor: '#e74c3c',
                    borderWidth: 1,
                    spanGaps: true
                }, {
                    label: this.text.charts.historicalMin,
                    data: minTemps,
                    backgroundColor: 'rgba(46, 204, 113, 0.8)',
                    borderColor: '#2ecc71',
                    borderWidth: 1,
                    spanGaps: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: this.text.charts.seasonalTitle,
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: this.text.charts.yAxisTemp
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: this.text.charts.xAxisMonth
                        }
                    }
                }
            }
        });
    }

    updateStatistics(stats) {
        const statsGrid = document.getElementById('statsGrid');
        if (!statsGrid) return;

        const statsInfo = stats.temperature_stats || {};
        const statsHTML = `
            <div class="stat-item">
                <div class="stat-value">${stats.total_days}</div>
                <div class="stat-label">${this.text.stats.totalDays}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${statsInfo.mean !== undefined ? statsInfo.mean + '°C' : '--'}</div>
                <div class="stat-label">${this.text.stats.avgTemp}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${statsInfo.min !== undefined ? statsInfo.min + '°C' : '--'}</div>
                <div class="stat-label">${this.text.stats.minTemp}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${statsInfo.max !== undefined ? statsInfo.max + '°C' : '--'}</div>
                <div class="stat-label">${this.text.stats.maxTemp}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${statsInfo.std !== undefined ? statsInfo.std + '°C' : '--'}</div>
                <div class="stat-label">${this.text.stats.stdTemp}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.date_range.start}</div>
                <div class="stat-label">${this.text.stats.startDate}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.date_range.end}</div>
                <div class="stat-label">${this.text.stats.endDate}</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${stats.current_temperature !== undefined ? stats.current_temperature + '°C' : '--'}</div>
                <div class="stat-label">${this.text.stats.currentTemp}</div>
            </div>
        `;

        statsGrid.innerHTML = statsHTML;
    }

    updatePerformanceMetrics(performanceData) {
        const metricsContainer = document.getElementById('performanceMetrics');
        if (!metricsContainer) return;

        const metrics = performanceData.metrics || {};
        const physicsMetrics = performanceData.physics_metrics || {};

        const metricsHTML = `
            <div class="metric-item">
                <span class="metric-label">${this.text.metrics.r2}</span>
                <span class="metric-value">${metrics.R2 !== undefined ? metrics.R2 : '--'}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">${this.text.metrics.rmse}</span>
                <span class="metric-value">${metrics.RMSE !== undefined ? metrics.RMSE + '°C' : '--'}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">${this.text.metrics.mae}</span>
                <span class="metric-value">${metrics.MAE !== undefined ? metrics.MAE + '°C' : '--'}</span>
            </div>
            <div class="metric-item">
                <span class="metric-label">${this.text.metrics.mape}</span>
                <span class="metric-value">${metrics.MAPE !== undefined ? metrics.MAPE + '%' : '--'}</span>
            </div>
            ${
                physicsMetrics.temperature_range_violation != null
                ? `<div class="metric-item">
                    <span class="metric-label">${this.text.metrics.tempViolation}</span>
                    <span class="metric-value">${(physicsMetrics.temperature_range_violation * 100).toFixed(1)}%</span>
                </div>`
                : ''
            }
            ${
                physicsMetrics.seasonal_consistency != null
                ? `<div class="metric-item">
                    <span class="metric-label">${this.text.metrics.seasonalConsistency}</span>
                    <span class="metric-value">${(physicsMetrics.seasonal_consistency * 100).toFixed(1)}%</span>
                </div>`
                : ''
            }
            ${
                physicsMetrics.smoothness_score != null
                ? `<div class="metric-item">
                    <span class="metric-label">${this.text.metrics.smoothness}</span>
                    <span class="metric-value">${(physicsMetrics.smoothness_score * 100).toFixed(1)}%</span>
                </div>`
                : ''
            }
            ${
                physicsMetrics.conservation_score != null
                ? `<div class="metric-item">
                    <span class="metric-label">${this.text.metrics.conservation}</span>
                    <span class="metric-value">${(physicsMetrics.conservation_score * 100).toFixed(1)}%</span>
                </div>`
                : ''
            }
            <div class="metric-item">
                <span class="metric-label">${this.text.metrics.dataSource}</span>
                <span class="metric-value">${this.currentData?.data_source || '--'}</span>
            </div>
        `;

        metricsContainer.innerHTML = metricsHTML;
    }

    getTemperatureStatus(temperature) {
        if (temperature < 10) return 'cold';
        if (temperature < 20) return 'cool';
        if (temperature < 25) return 'warm';
        return 'hot';
    }

    getTemperatureStatusText(temperature) {
        if (temperature < 10) return this.text.temperatureStatus.cold;
        if (temperature < 20) return this.text.temperatureStatus.cool;
        if (temperature < 25) return this.text.temperatureStatus.warm;
        return this.text.temperatureStatus.hot;
    }

    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = show ? 'flex' : 'none';
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.getElementById('notification');
        const notificationText = document.getElementById('notificationText');
        
        if (notification && notificationText) {
            notificationText.textContent = message;
            
            // 设置通知类型样式
            notification.className = `notification ${type}`;
            
            // 显示通知
            notification.classList.add('show');
            
            // 3秒后隐藏
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }
    }
}

// 全局函数，供HTML调用
function showSection(sectionId) {
    if (window.app) {
        window.app.showSection(sectionId);
    }
}

function makePrediction() {
    if (window.app) {
        window.app.makePrediction();
    }
}

// 页面加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    window.app = new TemperaturePredictionApp();
});

// 添加一些CSS样式到页面
const additionalStyles = `
    .status-badge {
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .status-badge.cold {
        background: #3498db;
        color: white;
    }
    
    .status-badge.cool {
        background: #2ecc71;
        color: white;
    }
    
    .status-badge.warm {
        background: #f39c12;
        color: white;
    }
    
    .status-badge.hot {
        background: #e74c3c;
        color: white;
    }
    
    .notification.error {
        background: #e74c3c;
    }
    
    .notification.success {
        background: #27ae60;
    }
    
    .notification.info {
        background: #3498db;
    }
`;

// 添加样式到页面
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);

