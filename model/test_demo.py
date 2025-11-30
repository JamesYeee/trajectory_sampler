import torch
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from model import FDMNet
from sklearn.preprocessing import StandardScaler
import random

# 设置设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"使用GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("使用CPU")

def load_model_and_scaler():
    """加载训练好的模型和标准化器"""
    # 加载模型
    model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = FDMNet(
        input_size=6,
        hidden_size=125,
        num_layers=3,
        output_size=18,
        sequence_length=5
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功!")
    
    # 加载标准化器
    scaler_path = os.path.join(os.path.dirname(__file__), 'feature_scaler.pkl')
    if not os.path.exists(scaler_path):
        print("警告: 未找到标准化器文件，将不进行特征标准化")
        return model, None
    
    with open(scaler_path, 'rb') as f:
        feature_scaler = pickle.load(f)
    print("特征标准化器加载成功!")
    
    return model, feature_scaler

def preprocess_csv_data(csv_file_path):
    """预处理CSV数据，与训练时保持一致"""
    print(f"处理CSV文件: {csv_file_path}")
    
    # 读取CSV数据
    df = pd.read_csv(csv_file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 数据预处理（与csv_to_train.py保持一致）
    window_size = 6
    step_size = 1
    min_brake = df['brake_sensor'].min()
    df['brake_sensor_scaled'] = (df['brake_sensor'] - min_brake) / (0.631 - min_brake)
    df['pedal_cc_scaled'] = df['pedal_cc'] / 1000
    
    # 定义特征和标签列
    feature_columns = ['vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    label_columns = ['vel_x', 'vel_y', 'rotation_rate']
    current_columns = ['pos_x', 'pos_y', 'heading', 'vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    
    features_list = []
    labels_list = []
    poses_list = []
    
    # 创建时间窗口
    num_windows = (len(df) - window_size) // step_size + 1
    print(f"可创建的时间窗口数量: {num_windows}")
    
    for i in range(num_windows):
        time_slice = df.iloc[i:i+window_size]
        
        # 特征: t-4 到 t 的5个时间步
        feature_slice = time_slice.iloc[:5][feature_columns]
        features_list.append(feature_slice.astype(np.float32).values)
        
        # 标签: t+1 时刻的速度状态
        label_slice = time_slice.iloc[5][label_columns]
        labels_list.append(label_slice.astype(np.float32).values)
        
        # 当前状态: t 时刻的完整状态
        current_slice = time_slice.iloc[4][current_columns]
        poses_list.append(current_slice.astype(np.float32).values)
    
    return np.array(features_list), np.array(labels_list), np.array(poses_list)

def apply_feature_normalization(features, scaler):
    """应用特征标准化"""
    if scaler is None:
        return features
    
    original_shape = features.shape
    features_reshaped = features.reshape(-1, original_shape[-1])
    features_normalized = scaler.transform(features_reshaped)
    return features_normalized.reshape(original_shape)

def test_specific_row(model, feature_scaler, csv_file_path, current_state_row=53):
    """测试指定原始行作为当前状态的下一个状态预测"""
    print(f"\n=== 以原始CSV第 {current_state_row} 行为当前状态，预测第 {current_state_row+1} 行 ===")
    
    # 预处理数据
    features_row, labels, poses = preprocess_csv_data(csv_file_path)
    
    if len(features_row) == 0:
        print("错误: 没有有效的数据样本")
        return None
    
    # 将原始行号转换为窗口索引
    # 当前状态在原始CSV第current_state_row行，对应窗口索引为current_state_row-4
    # 因为窗口i的当前状态是原始数据的第i+4行
    target_window_index = current_state_row - 4
    
    # 检查窗口索引是否有效
    if target_window_index < 0:
        print(f"错误: 原始行 {current_state_row} 太小，需要至少第4行才能作为当前状态")
        return None
    
    if target_window_index >= len(features_row):
        print(f"错误: 原始行 {current_state_row} 超出可用范围")
        print(f"最大可用的原始行号: {len(features_row) + 4 - 1}")
        return None
    
    # 应用标准化
    if feature_scaler is not None:
        features = apply_feature_normalization(features_row, feature_scaler)
        print("已应用特征标准化")
    else:
        features = features_row
    
    print(f"原始CSV当前状态行: {current_state_row}")
    print(f"预测目标行: {current_state_row + 1}")
    print(f"对应窗口索引: {target_window_index}")
    
    # 获取指定窗口的数据
    target_features = features[target_window_index:target_window_index+1]  # 保持batch维度
    target_label = labels[target_window_index]  # 真实的下一状态
    target_pose = poses[target_window_index:target_window_index+1]  # 保持batch维度
    
    # 转换为张量
    features_tensor = torch.FloatTensor(target_features).to(device)
    label_tensor = torch.FloatTensor([target_label]).to(device)
    pose_tensor = torch.FloatTensor(target_pose).to(device)
    
    # 模型预测
    model.eval()
    with torch.no_grad():
        prediction = model(features_tensor, pose_tensor)
    
    # 转换回numpy进行分析
    pred_values = prediction[0].cpu().numpy()
    true_values = target_label
    
    # 详细输出
    state_names = ['vel_x', 'vel_y', 'rotation_rate']
    
    print(f"\n=== 预测结果 (当前状态: 原始第{current_state_row}行) ===")
    print("真实值 -> 预测值 (绝对误差, 相对误差%):")
    
    total_error = 0
    for i, name in enumerate(state_names):
        true_val = float(true_values[i])
        pred_val = float(pred_values[i])
        abs_error = abs(pred_val - true_val)
        rel_error = (abs_error / abs(true_val) * 100) if abs(true_val) > 1e-6 else 0
        total_error += abs_error
        
        print(f"  {name}: {true_val:.6f} -> {pred_val:.6f} (误差: {abs_error:.6f}, {rel_error:.2f}%)")
    
    print(f"\n总绝对误差: {total_error:.6f}")
    
    # 显示输入信息
    print(f"\n=== 输入信息 ===")
    input_features = features_row[target_window_index]  # (5, 6) - 5个时间步，6个特征
    current_pose = poses[target_window_index]  # (9,) - 当前完整状态
    
    print(f"输入特征序列 (原始CSV第{current_state_row-4}到第{current_state_row}行):")
    feature_names = ['vel_x', 'vel_y', 'rotation_rate', 'pedal_cc', 'steer', 'brake']
    for t in range(5):
        csv_row = current_state_row - 4 + t
        print(f"  原始CSV第{csv_row}行:")
        for i, name in enumerate(feature_names):
            print(f"    {name}: {input_features[t, i]:.6f}")
    
    print(f"\n当前状态 (原始CSV第{current_state_row}行):")
    pose_names = ['pos_x', 'pos_y', 'heading', 'vel_x', 'vel_y', 'rotation_rate', 'pedal_cc', 'steer', 'brake']
    for i, name in enumerate(pose_names):
        print(f"  {name}: {current_pose[i]:.6f}")
    
    # 返回结果用于可视化
    result = {
        'current_state_row': current_state_row,
        'window_index': target_window_index,
        'true_values': true_values,
        'pred_values': pred_values,
        'input_features': input_features,
        'current_pose': current_pose,
        'total_error': total_error
    }
    
    return result



def main():
    """主函数"""
    print("=== FDM模型单行预测演示 ===")
    
    try:
        # 加载模型和标准化器
        model, feature_scaler = load_model_and_scaler()
        
        # CSV文件路径
        csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'FDM_dataset', 'scene-0069_merged_data.csv')
        
        if not os.path.exists(csv_file_path):
            print(f"错误: CSV文件不存在: {csv_file_path}")
            return
        
        # 指定原始CSV的行号作为当前状态
        current_state_row = 29  # 当前状态是原始CSV的第53行，预测第54行
        
        # 测试指定行
        test_specific_row(model, feature_scaler, csv_file_path, current_state_row)
        
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    main()