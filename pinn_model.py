"""
物理信息神经网络 (PINN) 模型
结合物理约束的海水温度预测模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
from config import *

class PhysicsInformedNN(nn.Module):
    """物理信息神经网络 — 含基于黄海 SST 物理特性的约束"""

    def __init__(self, input_dim, hidden_layers=None, activation='tanh',
                 feature_names=None, scaler_params=None):
        super(PhysicsInformedNN, self).__init__()

        if hidden_layers is None:
            hidden_layers = MODEL_CONFIG['hidden_layers']

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.feature_names = feature_names or []

        # 构建网络层
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'swish':
                layers.append(nn.SiLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        self._initialize_weights()
        self._resolve_feature_indices()
        self._setup_scaler_params(scaler_params)

    # -------------------------------------------------------------- #
    # 初始化辅助
    # -------------------------------------------------------------- #
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _resolve_feature_indices(self):
        """定位物理约束所需的特征列索引"""
        def _find(name):
            try:
                return self.feature_names.index(name)
            except ValueError:
                return None

        self.idx_doy_cos = _find('day_of_year_cos')
        self.idx_doy_sin = _find('day_of_year_sin')
        self.idx_lag1 = _find('mean_sst_lag_1')
        self.idx_lag2 = _find('mean_sst_lag_2')
        self.idx_lag7 = _find('mean_sst_lag_7')

    def _setup_scaler_params(self, params):
        """保存 StandardScaler/MinMaxScaler 参数，用于在物理空间做约束"""
        if params:
            self.register_buffer('sst_ss_mean',
                                 torch.tensor(params['ss_mean'], dtype=torch.float32))
            self.register_buffer('sst_ss_scale',
                                 torch.tensor(params['ss_scale'], dtype=torch.float32))
            self.register_buffer('sst_mm_min',
                                 torch.tensor(params['mm_min'], dtype=torch.float32))
            self.register_buffer('sst_mm_scale',
                                 torch.tensor(params['mm_scale'], dtype=torch.float32))
            self.has_scaler = True
        else:
            self.has_scaler = False

    def _lag_to_pred_space(self, lag_ss):
        """将 lag 特征从 StandardScaler 空间转换到 MinMaxScaler（预测）空间"""
        if not self.has_scaler:
            return lag_ss
        physical = lag_ss * self.sst_ss_scale + self.sst_ss_mean
        return (physical - self.sst_mm_min) / self.sst_mm_scale

    # -------------------------------------------------------------- #
    # 前向传播
    # -------------------------------------------------------------- #
    def forward(self, x):
        return self.network(x)

    # -------------------------------------------------------------- #
    # 物理约束总入口
    # -------------------------------------------------------------- #
    def physics_loss(self, x, y_pred, y_true):
        total = torch.tensor(0.0, device=y_pred.device)

        if PHYSICS_CONFIG['temperature_range']:
            total = total + PHYSICS_CONFIG.get('lambda_range', 1.0) * self._range_loss(y_pred)

        if PHYSICS_CONFIG['seasonal_cycle']:
            total = total + PHYSICS_CONFIG.get('lambda_seasonal', 0.1) * self._seasonal_loss(x, y_pred)

        if PHYSICS_CONFIG['spatial_smoothness']:
            total = total + PHYSICS_CONFIG.get('lambda_smooth', 0.01) * self._smoothness_loss(x, y_pred)

        if PHYSICS_CONFIG['conservation_law']:
            total = total + PHYSICS_CONFIG.get('lambda_conservation', 0.05) * self._conservation_loss(x, y_pred)

        return total

    # -------------------------------------------------------------- #
    # 约束 1 — 温度范围
    # 在 MinMaxScaler 空间 [0,1] 中，超出 [-margin, 1+margin] 的值受平方惩罚。
    # -------------------------------------------------------------- #
    def _range_loss(self, y_pred):
        margin = 0.05
        over = torch.relu(y_pred - (1.0 + margin))
        under = torch.relu(-(y_pred - (-margin)))
        return torch.mean(over ** 2 + under ** 2)

    # -------------------------------------------------------------- #
    # 约束 2 — 季节周期
    # 黄海 SST ≈ A − B·cos(2π·doy/365)：冬季 cos≈1 温低，夏季 cos≈−1 温高。
    # 强制预测值与 day_of_year_cos 呈负相关（Pearson r < −0.3）。
    # -------------------------------------------------------------- #
    def _seasonal_loss(self, x, y_pred):
        if self.idx_doy_cos is None:
            return torch.tensor(0.0, device=y_pred.device)

        pred = y_pred.squeeze()
        doy_cos = x[:, self.idx_doy_cos]

        pred_c = pred - pred.mean()
        cos_c = doy_cos - doy_cos.mean()
        cov = torch.mean(pred_c * cos_c)
        std_pred = torch.sqrt(torch.mean(pred_c ** 2) + 1e-8)
        std_cos = torch.sqrt(torch.mean(cos_c ** 2) + 1e-8)
        corr = cov / (std_pred * std_cos)

        return torch.relu(corr + 0.3)

    # -------------------------------------------------------------- #
    # 约束 3 — 时间平滑性（二阶差分 / 加速度惩罚）
    # accel = pred − 2·lag₁ + lag₂ 代表温度变化的"加速度"，应接近 0。
    # 使用 lag 特征，不依赖 batch 排序。
    # -------------------------------------------------------------- #
    def _smoothness_loss(self, x, y_pred):
        if self.idx_lag1 is None or self.idx_lag2 is None:
            return torch.tensor(0.0, device=y_pred.device)

        pred = y_pred.squeeze()
        lag1 = self._lag_to_pred_space(x[:, self.idx_lag1])
        lag2 = self._lag_to_pred_space(x[:, self.idx_lag2])

        accel = pred - 2.0 * lag1 + lag2
        return torch.mean(accel ** 2)

    # -------------------------------------------------------------- #
    # 约束 4 — 热惯性 / 守恒
    # 海洋混合层热容大，SST 日变化不超过 ~2 °C。
    # 超出阈值的 |pred − lag₁| 受平方惩罚。
    # -------------------------------------------------------------- #
    def _conservation_loss(self, x, y_pred):
        if self.idx_lag1 is None:
            return torch.tensor(0.0, device=y_pred.device)

        pred = y_pred.squeeze()
        lag1 = self._lag_to_pred_space(x[:, self.idx_lag1])

        daily_change = torch.abs(pred - lag1)

        max_change_c = PHYSICS_CONFIG.get('max_daily_change_celsius', 2.0)
        if self.has_scaler:
            threshold = max_change_c / self.sst_mm_scale
        else:
            threshold = 0.06

        excess = torch.relu(daily_change - threshold)
        return torch.mean(excess ** 2)

class PINNTrainer:
    """PINN训练器 — 含 warm-up 与梯度裁剪"""

    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=MODEL_CONFIG['learning_rate'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=20, factor=0.5
        )
        self.train_losses = []
        self.val_losses = []
        self.physics_losses = []
        self.warmup_epochs = PHYSICS_CONFIG.get('warmup_epochs', 30)

    def _physics_weight(self, epoch):
        """物理损失权重线性 warm-up"""
        if self.warmup_epochs <= 0 or epoch >= self.warmup_epochs:
            return 1.0
        return epoch / self.warmup_epochs

    def train_epoch(self, train_loader, epoch=0):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_physics_loss = 0.0
        pw = self._physics_weight(epoch)

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            y_pred = self.model(batch_x)

            data_loss = nn.MSELoss()(y_pred.squeeze(), batch_y)
            physics_loss = self.model.physics_loss(batch_x, y_pred, batch_y)

            total_loss_batch = data_loss + pw * physics_loss

            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += data_loss.item()
            total_physics_loss += physics_loss.item()

        return total_loss / len(train_loader), total_physics_loss / len(train_loader)
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                y_pred = self.model(batch_x)
                loss = nn.MSELoss()(y_pred.squeeze(), batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, epochs=None):
        """训练模型"""
        if epochs is None:
            epochs = MODEL_CONFIG['epochs']
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"开始训练PINN模型，设备: {self.device}")
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"验证样本数: {len(val_loader.dataset)}")
        
        for epoch in range(epochs):
            train_loss, physics_loss = self.train_epoch(train_loader, epoch=epoch)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.physics_losses.append(physics_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  训练损失: {train_loss:.6f}")
                print(f"  验证损失: {val_loss:.6f}")
                print(f"  物理损失: {physics_loss:.6f}")
                print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.8f}")
            
            # 早停
            if patience_counter >= MODEL_CONFIG['early_stopping_patience']:
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        print("训练完成!")
        return self.train_losses, self.val_losses, self.physics_losses
    
    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(OUTPUT_CONFIG['model_save_path'], filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'physics_losses': self.physics_losses
        }, model_path)
    
    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join(OUTPUT_CONFIG['model_save_path'], filename)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.physics_losses = checkpoint['physics_losses']
    
    def plot_training_history(self, save_path='plots/'):
        """绘制训练历史"""
        plt.figure(figsize=(15, 5))
        
        # 训练和验证损失
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='训练损失')
        plt.plot(self.val_losses, label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.yscale('log')
        
        # 物理损失
        plt.subplot(1, 3, 2)
        plt.plot(self.physics_losses, label='物理损失', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('物理损失')
        plt.title('物理约束损失')
        plt.legend()
        plt.yscale('log')
        
        # 学习率
        plt.subplot(1, 3, 3)
        lr_history = [group['lr'] for group in self.optimizer.param_groups]
        plt.plot(lr_history, label='学习率', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('学习率')
        plt.title('学习率变化')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

class PINNPredictor:
    """PINN预测器"""
    
    def __init__(self, model, scaler, temperature_scaler):
        self.model = model
        self.scaler = scaler
        self.temperature_scaler = temperature_scaler
        self.model.eval()
    
    def predict(self, X):
        """预测"""
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            # 反归一化
            predictions = self.temperature_scaler.inverse_transform(
                predictions.numpy().reshape(-1, 1)
            ).flatten()
            return predictions
    
    def predict_sequence(self, initial_data, steps):
        """序列预测"""
        predictions = []
        current_data = initial_data.copy()
        
        for step in range(steps):
            pred = self.predict(current_data.reshape(1, -1))[0]
            predictions.append(pred)
            
            # 更新输入数据 (滑动窗口)
            current_data = np.roll(current_data, -1)
            current_data[-1] = pred
        
        return np.array(predictions)

def create_data_loaders(X_train, X_test, y_train, y_test, batch_size=None):
    """创建数据加载器"""
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    # 转换为张量
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.FloatTensor(y_train)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # 创建数据集
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def main():
    """主函数 - PINN模型训练"""
    print("请先运行 data_preprocessing.py 来准备数据")
    print("然后使用以下代码训练PINN模型:")
    
    example_code = """
    # 示例使用代码
    from data_preprocessing import main as preprocess_main
    import torch
    
    # 1. 数据预处理
    df, X_train, X_test, y_train, y_test, feature_columns = preprocess_main()
    
    # 2. 创建数据加载器
    train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test)
    
    # 3. 创建PINN模型
    model = PhysicsInformedNN(input_dim=len(feature_columns))
    
    # 4. 训练模型
    trainer = PINNTrainer(model)
    train_losses, val_losses, physics_losses = trainer.train(train_loader, test_loader)
    
    # 5. 绘制训练历史
    trainer.plot_training_history()
    
    # 6. 预测
    predictor = PINNPredictor(model, scaler, temperature_scaler)
    predictions = predictor.predict(X_test)
    """
    
    print(example_code)

if __name__ == "__main__":
    main()










