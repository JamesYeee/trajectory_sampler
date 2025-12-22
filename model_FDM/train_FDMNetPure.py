import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import FDMNetPure
import matplotlib.pyplot as plt
import os
import yaml
import pickle
from sklearn.preprocessing import StandardScaler


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("CPU")

def load_config():
    """加载训练配置"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'model_params', 'FDM_params.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data():
    """加载训练、验证和测试数据"""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # 加载数据
    train_data = np.load(os.path.join(data_dir, 'training_data.npz'))
    val_data = np.load(os.path.join(data_dir, 'validation_data.npz'))
    test_data = np.load(os.path.join(data_dir, 'testing_data.npz'))
    
    return train_data, val_data, test_data

def compute_feature_scaler(train_data):
    """计算features的标准化参数"""
    # train_data['features'] shape: (N, 5, 6)
    # 将所有时间步的特征展平用于计算统计量
    features = train_data['features']  # (N, 5, 6)
    
    # 重塑为 (N*5, 6) 以计算每个特征维度的统计量
    features_reshaped = features.reshape(-1, features.shape[-1])
    
    # 创建标准化器
    scaler = StandardScaler()
    scaler.fit(features_reshaped)
    
    print("Features标准化参数:")
    print(f"Mean: {scaler.mean_}")
    print(f"Std: {scaler.scale_}")
    
    return scaler

def apply_feature_normalization(features, scaler):
    """应用features标准化"""
    # features shape: (N, 5, 6)
    original_shape = features.shape
    
    # 重塑为 (N*5, 6)
    features_reshaped = features.reshape(-1, original_shape[-1])
    
    # 应用标准化
    features_normalized = scaler.transform(features_reshaped)
    
    # 重塑回原始形状
    features_normalized = features_normalized.reshape(original_shape)
    
    return features_normalized

def prepare_data(train_data, val_data, config, feature_scaler=None):
    """准备训练数据"""
    batch_size = config['TRAIN']['batch_size']
    
    # 提取数据
    X_train = train_data['features']  # (N, 5, 6)
    y_train = train_data['labels']    # (N, 3)
    poses_train = train_data['poses'] # (N, 9)
    
    X_val = val_data['features']
    y_val = val_data['labels']
    poses_val = val_data['poses']
    
    # 应用标准化
    if feature_scaler is not None:
        print("应用features标准化...")
        X_train = apply_feature_normalization(X_train, feature_scaler)
        X_val = apply_feature_normalization(X_val, feature_scaler)
    
    # 转换为张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    poses_train = torch.FloatTensor(poses_train)
    
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    poses_val = torch.FloatTensor(poses_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train, poses_train)
    val_dataset = TensorDataset(X_val, y_val, poses_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def weighted_mse_loss(predictions, targets, weights):
    """
    加权MSE损失函数
    
    Args:
        predictions: 模型预测值 (batch_size, 3)
        targets: 真实值 (batch_size, 3)
        weights: 权重 (3,) - [vel_x_weight, vel_y_weight, rotation_rate_weight]
    """
    # 计算每个维度的MSE
    mse_per_dim = torch.mean((predictions - targets) ** 2, dim=0)  # (3,)
    
    # 应用权重
    weighted_mse = torch.sum(mse_per_dim * weights)
    
    return weighted_mse

def train_model():
    """训练模型主函数"""
    # 加载配置
    config = load_config()

    # 从配置文件读取权重
    if 'loss_weights' in config['TRAIN']:
        weights_config = config['TRAIN']['loss_weights']
        loss_weights = torch.tensor([
            weights_config['vel_x'],
            weights_config['vel_y'], 
            weights_config['rotation_rate']
        ], device=device)
    else:
        # 默认权重
        loss_weights = torch.tensor([1.0, 1.0, 5.0], device=device)
    
    print(f"损失权重: vel_x={loss_weights[0]}, vel_y={loss_weights[1]}, rotation_rate={loss_weights[2]}")
    
    # 加载数据
    train_data, val_data, test_data = load_data()
    
    # 计算features标准化参数
    feature_scaler = compute_feature_scaler(train_data)
    
    # 保存标准化器（使用不同的文件名以区分）
    scaler_path = os.path.join(os.path.dirname(__file__), 'feature_scaler_pure.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)
    print(f"标准化器已保存至: {scaler_path}")
    
    # 准备数据（应用标准化）
    train_loader, val_loader = prepare_data(train_data, val_data, config, feature_scaler)
    
    print(f"训练样本数: {len(train_data['features'])}")
    print(f"验证样本数: {len(val_data['features'])}")
    print(f"测试样本数: {len(test_data['features'])}")
    
    # 创建FDMNetPure模型
    model = FDMNetPure(
        input_size=6,
        hidden_size=125,  
        num_layers=3,
        output_size=3,   
        sequence_length=5
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])
    
    # 训练参数
    epochs = config['TRAIN']['epochs']
    best_val_loss = float('inf')
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    
    print("\n开始训练FDMNetPure...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (features, labels, poses) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device) 
            poses = poses.to(device)
            
            optimizer.zero_grad()
            # FDMNetPure直接输出状态预测，不使用poses（但保持API兼容性）
            outputs = model(features, poses)
            
            loss = weighted_mse_loss(outputs, labels, loss_weights)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels, poses in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                poses = poses.to(device)
                
                outputs = model(features, poses)
                loss = weighted_mse_loss(outputs, labels, loss_weights)
                val_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        
        # 保存最佳模型（使用不同的文件名）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'best_model_pure.pth')
    
    print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('FDMNetPure Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves_pure.png')
    print("训练曲线已保存至: training_curves_pure.png")
    
    return model, train_losses, val_losses

def test_model():
    """测试模型"""
    # 加载配置以获取权重
    config = load_config()
    if 'loss_weights' in config['TRAIN']:
        weights_config = config['TRAIN']['loss_weights']
        loss_weights = torch.tensor([
            weights_config['vel_x'],
            weights_config['vel_y'], 
            weights_config['rotation_rate']
        ], device=device)
    else:
        loss_weights = torch.tensor([1.0, 1.0, 5.0], device=device)
    
    # 加载标准化器
    scaler_path = os.path.join(os.path.dirname(__file__), 'feature_scaler_pure.pkl')
    try:
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print("已加载features标准化器")
    except FileNotFoundError:
        print("警告: 未找到标准化器文件，将不进行标准化")
        feature_scaler = None
    
    # 加载最佳模型
    checkpoint = torch.load('best_model_pure.pth')
    
    model = FDMNetPure(
        input_size=6,
        hidden_size=125,
        num_layers=3,
        output_size=3,
        sequence_length=5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"加载模型 (Epoch {checkpoint['epoch']+1})")
    
    # 加载测试数据
    _, _, test_data = load_data()
    
    X_test = test_data['features']
    y_test = test_data['labels']
    poses_test = test_data['poses']
    
    # 应用标准化
    if feature_scaler is not None:
        X_test = apply_feature_normalization(X_test, feature_scaler)
    
    # 转换为张量
    X_test = torch.FloatTensor(X_test).to(device)
    y_test = torch.FloatTensor(y_test).to(device)
    poses_test = torch.FloatTensor(poses_test).to(device)
    
    # 测试
    with torch.no_grad():
        predictions = model(X_test, poses_test)
        
        # 计算加权测试损失
        test_loss = weighted_mse_loss(predictions, y_test, loss_weights)
        print(f"\n加权测试损失: {test_loss.item():.6f}")
        
        # 计算标准MSE损失用于比较
        standard_mse = nn.MSELoss()(predictions, y_test)
        print(f"标准MSE损失: {standard_mse.item():.6f}")
        
        # 计算各个状态变量的MAE和MSE
        mae = torch.mean(torch.abs(predictions - y_test), dim=0)
        mse = torch.mean((predictions - y_test) ** 2, dim=0)
        state_names = ['vel_x', 'vel_y', 'rotation_rate']
        
        print("\n各状态变量的误差统计:")
        print("-" * 60)
        print(f"{'Variable':<20} {'MAE':<15} {'MSE':<15}")
        print("-" * 60)
        for i, name in enumerate(state_names):
            print(f"{name:<20} {mae[i].item():<15.6f} {mse[i].item():<15.6f}")
        print("-" * 60)

if __name__ == "__main__":
    # 训练模型
    model, train_losses, val_losses = train_model()
    
    # 测试模型
    test_model()