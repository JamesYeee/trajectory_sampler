import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yaml
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model_sampler import Simplified_Trajectory_Sampler, compute_cvae_loss


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def compute_feature_scalers(waypoints, actions):
    """计算waypoints和actions的标准化参数"""
    # waypoints shape: (N, 4, 3) -> (N*4, 3)
    waypoints_reshaped = waypoints.reshape(-1, waypoints.shape[-1])
    waypoint_scaler = StandardScaler()
    waypoint_scaler.fit(waypoints_reshaped)
    
    # actions shape: (N, 20, 3) -> (N*20, 3)  
    actions_reshaped = actions.reshape(-1, actions.shape[-1])
    action_scaler = StandardScaler()
    action_scaler.fit(actions_reshaped)
    
    print("Waypoints标准化参数:")
    print(f"Mean: {waypoint_scaler.mean_}")
    print(f"Std: {waypoint_scaler.scale_}")
    
    print("Actions标准化参数:")
    print(f"Mean: {action_scaler.mean_}")
    print(f"Std: {action_scaler.scale_}")
    
    return waypoint_scaler, action_scaler


def apply_feature_normalization(data, scaler):
    """应用特征标准化"""
    original_shape = data.shape
    # 重塑为 (N*seq_len, feature_dim)
    data_reshaped = data.reshape(-1, original_shape[-1])
    # 应用标准化
    data_normalized = scaler.transform(data_reshaped)
    # 重塑回原始形状
    return data_normalized.reshape(original_shape)


def load_and_split_data(data_path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    加载并划分数据集
    
    Args:
        data_path: 数据文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
    
    Returns:
        train_data, val_data, test_data, scalers
    """
    # 加载数据
    data = np.load(data_path)
    conditions = data['conditions']  # waypoints: (N, 4, 3) -> pos_x, pos_y, heading
    labels = data['labels']         # actions: (N, 20, 3) -> pedal_cc_scaled, steer_corrected, brake_sensor_scaled
    
    print(f"Data loaded: conditions shape {conditions.shape}, labels shape {labels.shape}")
    
    # 提取位置信息作为waypoints (只使用pos_x, pos_y)
    waypoints = conditions  # (N, 4, 3)
    actions = labels  # (N, 20, 3)
    
    # 划分数据集
    # 首先划分训练集和临时集合(验证+测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        waypoints, actions, 
        test_size=(val_ratio + test_ratio), 
        random_state=42
    )
    
    # 然后划分验证集和测试集
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(1 - val_size), 
        random_state=42
    )
    
    print(f"Dataset split:")
    print(f"Train: {len(X_train)} samples")
    print(f"Validation: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # 计算标准化参数（只用训练集）
    waypoint_scaler, action_scaler = compute_feature_scalers(X_train, y_train)
    
    # 应用标准化
    X_train_norm = apply_feature_normalization(X_train, waypoint_scaler)
    X_val_norm = apply_feature_normalization(X_val, waypoint_scaler)
    X_test_norm = apply_feature_normalization(X_test, waypoint_scaler)
    
    y_train_norm = apply_feature_normalization(y_train, action_scaler)
    y_val_norm = apply_feature_normalization(y_val, action_scaler)
    y_test_norm = apply_feature_normalization(y_test, action_scaler)
    
    return (
        (X_train_norm, y_train_norm), 
        (X_val_norm, y_val_norm), 
        (X_test_norm, y_test_norm),
        (waypoint_scaler, action_scaler)
    )


def create_data_loaders(train_data, val_data, test_data, batch_size):
    """创建数据加载器"""
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data
    
    # 转换为torch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, device, kl_weight=0.001):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    for batch_idx, (waypoints, actions) in enumerate(train_loader):
        waypoints = waypoints.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        mu, logvar, reconstructed_actions, sampled_actions = model(
            waypoints, actions, training=True
        )
        
        # 计算损失
        loss, recon_loss, kl_loss = compute_cvae_loss(
            mu, logvar, reconstructed_actions, actions, kl_weight
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def validate_epoch(model, val_loader, device, kl_weight=0.001):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    
    with torch.no_grad():
        for waypoints, actions in val_loader:
            waypoints = waypoints.to(device)
            actions = actions.to(device)
            
            # 前向传播
            mu, logvar, reconstructed_actions, sampled_actions = model(
                waypoints, actions, training=True
            )
            
            # 计算损失
            loss, recon_loss, kl_loss = compute_cvae_loss(
                mu, logvar, reconstructed_actions, actions, kl_weight
            )
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_kl_loss = total_kl_loss / len(val_loader)
    
    return avg_loss, avg_recon_loss, avg_kl_loss


def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 总损失
    plt.subplot(1, 3, 1)
    plt.plot(epochs, [loss[0] for loss in train_losses], 'b-', label='Train Total Loss')
    plt.plot(epochs, [loss[0] for loss in val_losses], 'r-', label='Val Total Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 重构损失
    plt.subplot(1, 3, 2)
    plt.plot(epochs, [loss[1] for loss in train_losses], 'b-', label='Train Recon Loss')
    plt.plot(epochs, [loss[1] for loss in val_losses], 'r-', label='Val Recon Loss')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # KL损失
    plt.subplot(1, 3, 3)
    plt.plot(epochs, [loss[2] for loss in train_losses], 'b-', label='Train KL Loss')
    plt.plot(epochs, [loss[2] for loss in val_losses], 'r-', label='Val KL Loss')
    plt.title('KL Divergence Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def main():
    # 加载配置
    config_path = '../model_params/sampler_params.yaml'
    config = load_config(config_path)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载和划分数据
    data_path = '../data/sampler_dataset.npz'
    train_data, val_data, test_data, scalers = load_and_split_data(data_path)
    waypoint_scaler, action_scaler = scalers
    
    # 保存标准化器
    scaler_dir = os.path.dirname(__file__)
    waypoint_scaler_path = os.path.join(scaler_dir, 'waypoint_scaler.pkl')
    action_scaler_path = os.path.join(scaler_dir, 'action_scaler.pkl')
    
    with open(waypoint_scaler_path, 'wb') as f:
        pickle.dump(waypoint_scaler, f)
    with open(action_scaler_path, 'wb') as f:
        pickle.dump(action_scaler, f)
    
    print(f"Scalers saved to {waypoint_scaler_path} and {action_scaler_path}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, 
        batch_size=config['TRAIN']['batch_size']
    )
    
    # 创建模型
    model = Simplified_Trajectory_Sampler(
    waypoint_dim=config['MODEL']['waypoint_dim'],
    action_dim=config['MODEL']['action_dim'],
    latent_dim=config['MODEL']['latent_dim'],
    hidden_dims=config['MODEL']['hidden_dims'],
    device=device
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=config['TRAIN']['lr'])
    
    # 训练参数
    num_epochs = config['TRAIN']['epochs']
    kl_weight = config['TRAIN']['kl_weight']
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_recon_loss, train_kl_loss = train_epoch(
            model, train_loader, optimizer, device, kl_weight
        )
        
        # 验证
        val_loss, val_recon_loss, val_kl_loss = validate_epoch(
            model, val_loader, device, kl_weight
        )
        
        # 记录损失
        train_losses.append((train_loss, train_recon_loss, train_kl_loss))
        val_losses.append((val_loss, val_recon_loss, val_kl_loss))
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train - Total: {train_loss:.4f}, Recon: {train_recon_loss:.4f}, KL: {train_kl_loss:.4f}')
        print(f'Val   - Total: {val_loss:.4f}, Recon: {val_recon_loss:.4f}, KL: {val_kl_loss:.4f}')
        print('-' * 60)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_sampler_model.pth')
            print(f'Best model saved at epoch {epoch+1}')
    
    # 保存最终模型
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'final_sampler_model.pth')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, 'training_curves.png')
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()