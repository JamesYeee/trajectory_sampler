from model_sampler import Simplified_Trajectory_Sampler
import torch
import numpy as np
import pandas as pd
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 添加 model 目录到路径
sys.path.append('/home/jamesye/semester_arbeit/FDM/model')
from model import FDMNet

def apply_feature_normalization(data, scaler):
    """应用特征标准化"""
    if scaler is None:
        return data
    
    original_shape = data.shape
    # 重塑为 (N*seq_len, feature_dim)
    data_reshaped = data.reshape(-1, original_shape[-1])
    # 应用标准化
    data_normalized = scaler.transform(data_reshaped)
    # 重塑回原始形状
    return data_normalized.reshape(original_shape)

def denormalize_actions(normalized_actions, action_scaler):
    """
    将标准化后的actions反标准化为真实值
    
    Args:
        normalized_actions: (batch_size, 20, 3) - 标准化的actions
        action_scaler: sklearn StandardScaler对象
    
    Returns:
        real_actions: (batch_size, 20, 3) - 真实的actions
    """
    original_shape = normalized_actions.shape
    # 重塑为 (batch_size*20, 3)
    actions_reshaped = normalized_actions.reshape(-1, original_shape[-1])
    
    # 确保数据类型和内存连续性
    actions_reshaped = np.ascontiguousarray(actions_reshaped, dtype=np.float64)
    
    # 反标准化: x_real = x_normalized * std + mean
    real_actions = action_scaler.inverse_transform(actions_reshaped)
    
    # 重塑回原始形状
    return real_actions.reshape(original_shape)

def load_sampler_and_scalers():
    """加载训练好的sampler模型和标准化器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载sampler模型
    sampler = Simplified_Trajectory_Sampler(
        waypoint_dim=3,  # pos_x, pos_y, heading
        action_dim=3,    # pedal_cc, steer_corrected, brake_sensor
        latent_dim=64,
        hidden_dims=[64, 64],
        device=device
    ).to(device)
    
    # 加载sampler模型权重
    sampler_model_path = '/home/jamesye/semester_arbeit/FDM/model_sampler/best_sampler_model.pth'
    checkpoint = torch.load(sampler_model_path, map_location=device)
    sampler.load_state_dict(checkpoint['model_state_dict'])
    sampler.eval()
    
    # 加载sampler标准化器
    waypoint_scaler_path = '/home/jamesye/semester_arbeit/FDM/model_sampler/waypoint_scaler.pkl'
    action_scaler_path = '/home/jamesye/semester_arbeit/FDM/model_sampler/action_scaler.pkl'
    
    with open(waypoint_scaler_path, 'rb') as f:
        waypoint_scaler = pickle.load(f)
    with open(action_scaler_path, 'rb') as f:
        action_scaler = pickle.load(f)
    
    return sampler, waypoint_scaler, action_scaler, device

def load_fdm_model_and_scaler():
    """加载训练好的FDM模型和特征标准化器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载FDM模型
    fdm_model = FDMNet(
        input_size=6,
        hidden_size=125,
        num_layers=3,
        output_size=18,
        sequence_length=5
    ).to(device)
    
    # 加载FDM模型权重
    fdm_model_path = '/home/jamesye/semester_arbeit/FDM/model/best_model.pth'
    checkpoint = torch.load(fdm_model_path, map_location=device)
    fdm_model.load_state_dict(checkpoint['model_state_dict'])
    fdm_model.eval()
    
    # 加载特征标准化器
    feature_scaler_path = '/home/jamesye/semester_arbeit/FDM/model/feature_scaler.pkl'
    try:
        with open(feature_scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print("Feature scaler loaded successfully")
    except FileNotFoundError:
        print(f"Warning: Feature scaler file not found at {feature_scaler_path}")
        feature_scaler = None
    
    return fdm_model, feature_scaler, device

def preprocess_csv_data(csv_file_path):
    """预处理CSV数据，与训练保持一致"""
    df = pd.read_csv(csv_file_path)
    
    # 数据预处理 - 与训练保持一致
    min_brake = df['brake_sensor'].min()
    df['brake_sensor_scaled'] = (df['brake_sensor'] - min_brake) / (0.631 - min_brake)
    df['pedal_cc_scaled'] = df['pedal_cc'] / 1000
    
    return df

def extract_waypoints_from_csv(df, current_idx=29):
    """
    从CSV文件中提取waypoints
    
    Args:
        df: DataFrame
        current_idx: 当前状态的索引
    
    Returns:
        waypoints: (1, 4, 3) - 未来2秒的waypoints
        current_state: 当前状态信息
    """
    # 检查索引是否有效
    if current_idx >= len(df) - 20:  # 确保有足够的未来数据
        raise ValueError(f"current_idx={current_idx} 太大，CSV只有 {len(df)} 行数据")
    
    # 当前状态
    current_state = df.iloc[current_idx]
    print(f"当前状态 (idx={current_idx}):")
    print(f"  时间: {current_state['utime']}")
    print(f"  位置: ({current_state['pos_x']:.3f}, {current_state['pos_y']:.3f})")
    print(f"  航向: {current_state['heading']:.3f}")
    print(f"  速度: ({current_state['vel_x']:.3f}, {current_state['vel_y']:.3f})")
    
    # 提取未来2秒的waypoints (2Hz × 2秒 = 4个点)
    # 假设数据是10Hz采集，所以每隔5个点取一个waypoint
    waypoint_indices = [current_idx + 5*i for i in range(1, 5)]  # idx+5, idx+10, idx+15, idx+20
    
    waypoints = []
    for idx in waypoint_indices:
        if idx < len(df):
            row = df.iloc[idx]
            waypoint = [row['pos_x'], row['pos_y'], row['heading']]
            waypoints.append(waypoint)
        else:
            # 如果超出数据范围，使用最后一个有效点
            waypoints.append(waypoints[-1] if waypoints else [0, 0, 0])
    
    waypoints = np.array(waypoints).reshape(1, 4, 3)  # (1, 4, 3)
    
    print(f"提取的waypoints:")
    for i, wp in enumerate(waypoints[0]):
        print(f"  Waypoint {i+1}: pos=({wp[0]:.3f}, {wp[1]:.3f}), heading={wp[2]:.3f}")
    
    return waypoints, current_state

def generate_multiple_actions_with_sampler(sampler, waypoints, waypoint_scaler, action_scaler, device, n_samples=10):
    """
    使用sampler生成多条actions轨迹
    
    Args:
        sampler: 训练好的sampler模型
        waypoints: (1, 4, 3) - waypoints
        waypoint_scaler: waypoint标准化器
        action_scaler: action标准化器
        device: 设备
        n_samples: 生成的轨迹数量
    
    Returns:
        real_actions_list: List of (1, 20, 3) - 多条真实的action序列
    """
    # 1. waypoints预处理（标准化）
    normalized_waypoints = apply_feature_normalization(waypoints, waypoint_scaler)
    normalized_waypoints = torch.FloatTensor(normalized_waypoints).to(device)
    
    # 2. 生成多条actions轨迹
    sampler.eval()
    with torch.no_grad():
        # 使用sample_multiple方法生成多条轨迹
        multiple_normalized_actions = sampler.sample_multiple(normalized_waypoints, n_samples=n_samples)  # (1, n_samples, 20, 3)
    
    # 3. actions后处理（反标准化）
    multiple_normalized_actions_np = multiple_normalized_actions.cpu().numpy()  # (1, n_samples, 20, 3)
    
    real_actions_list = []
    for i in range(n_samples):
        # 提取第i条轨迹的actions: (1, 20, 3)
        single_normalized_actions = multiple_normalized_actions_np[:, i, :, :]  # (1, 20, 3)
        real_actions = denormalize_actions(single_normalized_actions, action_scaler)
        real_actions_list.append(real_actions)
        
        print(f"\n=== 轨迹 {i+1} 的actions序列 (前5步) ===")
        for j in range(min(5, real_actions.shape[1])):
            action = real_actions[0, j]
            print(f"  Step {j+1}: pedal={action[0]:.3f}, steer={action[1]:.3f}, brake={action[2]:.3f}")
    
    return real_actions_list

def create_sequence_from_start(df, start_idx, sequence_length=5, feature_scaler=None):
    """从起始点创建输入序列，应用特征标准化"""
    if start_idx + sequence_length > len(df):
        return None, None
    
    # 特征列 - 与训练保持一致
    feature_columns = ['vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    current_columns = ['pos_x', 'pos_y', 'heading', 'vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    
    # 提取序列特征 (t-4 to t)
    sequence_data = df.iloc[start_idx:start_idx + sequence_length]
    features = sequence_data[feature_columns].values.astype(np.float32)
    
    # 应用特征标准化
    if feature_scaler is not None:
        features = apply_feature_normalization(features, feature_scaler)
        print("Applied feature normalization to input sequence")
    
    # 当前状态 (at time t) - poses不标准化
    current_state = df.iloc[start_idx + sequence_length - 1][current_columns].values.astype(np.float32)
    
    return features, current_state

def predict_multiple_trajectories_with_sampler_actions(fdm_model, df, multiple_sampler_actions, start_idx, feature_scaler=None, prediction_steps=20):
    """
    使用多条sampler生成的actions进行轨迹预测
    
    Args:
        fdm_model: 训练好的FDM模型
        df: DataFrame
        multiple_sampler_actions: List of (1, 20, 3) - 多条sampler生成的actions
        start_idx: 起始索引
        feature_scaler: 特征标准化器
        prediction_steps: 预测步数
    
    Returns:
        trajectories: List of trajectories
        ground_truth: 真实轨迹 (如果可用)
        start_info: 起始点信息
        all_states: List of states for each trajectory
    """
    print(f"使用 {len(multiple_sampler_actions)} 条sampler actions从索引 {start_idx} 开始预测轨迹")
    
    trajectories = []
    all_states = []
    
    # 获取真实轨迹 (只需要计算一次)
    ground_truth = []
    ground_truth_states = []
    if start_idx + prediction_steps < len(df):
        for i in range(prediction_steps + 1):
            gt_pos_x = df.iloc[start_idx + i]['pos_x']
            gt_pos_y = df.iloc[start_idx + i]['pos_y']
            ground_truth.append([gt_pos_x, gt_pos_y])

            # 获取真实状态
            gt_vel_x = df.iloc[start_idx + i]['vel_x']
            gt_vel_y = df.iloc[start_idx + i]['vel_y']
            gt_rotation_rate = df.iloc[start_idx + i]['rotation_rate']
            ground_truth_states.append([gt_vel_x, gt_vel_y, gt_rotation_rate])
    
    # 起始点信息
    features, current_state = create_sequence_from_start(df, start_idx - 5, 5, feature_scaler)
    if features is None:
        raise ValueError("无法创建有效的序列数据")
    
    start_info = {
        'index': start_idx,
        'time': df.iloc[start_idx]['utime'],
        'pos_x': current_state[0],
        'pos_y': current_state[1],
        'vel_x': current_state[3],
        'vel_y': current_state[4]
    }
    
    # 为每条action序列预测轨迹
    for traj_idx, sampler_actions in enumerate(multiple_sampler_actions):
        print(f"\n--- 预测轨迹 {traj_idx + 1} ---")
        
        # 创建初始序列 (每条轨迹都从相同起始点开始)
        features, current_state = create_sequence_from_start(df, start_idx - 5, 5, feature_scaler)
        
        device = next(fdm_model.parameters()).device
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # (1, 5, 6)
        current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)  # (1, 9)
        
        # 存储预测轨迹
        trajectory = []
        states = []
        
        # 初始状态
        current_pos_x = current_state[0]
        current_pos_y = current_state[1] 
        current_heading = current_state[2]
        current_vel_x = current_state[3]
        current_vel_y = current_state[4]
        current_rotation_rate = current_state[5]
        
        trajectory.append([current_pos_x, current_pos_y])
        states.append([current_vel_x, current_vel_y, current_rotation_rate])
        
        with torch.no_grad():
            for step in range(prediction_steps):
                # 模型预测下一步的速度状态
                pred_states = fdm_model(features_tensor, current_state_tensor)  # (1, 3)
                
                # 提取预测状态
                pred_vel_x = pred_states[0, 0].item()
                pred_vel_y = 0  # pred_states[0, 1].item()
                pred_rotation_rate = pred_states[0, 2].item()
                
                # 使用sampler生成的actions
                sampler_action = sampler_actions[0, step]  # (3,)
                pedal_cc_scaled = sampler_action[0]
                steer_corrected = sampler_action[1] 
                brake_sensor_scaled = sampler_action[2]
                
                # 积分获得位置和航向 (dt = 0.1s)
                dt = 0.1
                next_pos_x = current_pos_x + dt * (current_vel_x * np.cos(current_heading) - current_vel_y * np.sin(current_heading))
                next_pos_y = current_pos_y + dt * (current_vel_x * np.sin(current_heading) + current_vel_y * np.cos(current_heading))
                next_heading = current_heading + dt * current_rotation_rate
                
                trajectory.append([next_pos_x, next_pos_y])
                states.append([pred_vel_x, pred_vel_y, pred_rotation_rate])

                # 更新状态
                current_pos_x = next_pos_x
                current_pos_y = next_pos_y
                current_heading = next_heading
                current_vel_x = pred_vel_x
                current_vel_y = pred_vel_y
                current_rotation_rate = pred_rotation_rate
                
                # 更新输入序列 - 滑动窗口
                new_feature = np.array([pred_vel_x, pred_vel_y, pred_rotation_rate, 
                                      pedal_cc_scaled, steer_corrected, brake_sensor_scaled], dtype=np.float32)
                
                # 对新特征向量应用标准化
                if feature_scaler is not None:
                    new_feature = apply_feature_normalization(new_feature, feature_scaler)
                
                # 更新特征序列 (滑动窗口)
                features_tensor = torch.cat([features_tensor[:, 1:, :], 
                                           torch.FloatTensor(new_feature).unsqueeze(0).unsqueeze(0).to(device)], dim=1)
                
                # 更新当前状态张量
                current_state_tensor = torch.FloatTensor([
                    current_pos_x, current_pos_y, current_heading,
                    current_vel_x, current_vel_y, current_rotation_rate,
                    pedal_cc_scaled, steer_corrected, brake_sensor_scaled
                ]).unsqueeze(0).to(device)
        
        trajectories.append(np.array(trajectory))
        all_states.append(np.array(states))
    
    return (trajectories, 
            np.array(ground_truth) if ground_truth else None, 
            start_info, 
            all_states,
            np.array(ground_truth_states) if ground_truth_states else None)

def plot_multiple_trajectories(trajectories, ground_truth=None, start_info=None, save_path='multiple_sampler_trajectories.png'):
    """绘制多条预测轨迹"""
    plt.figure(figsize=(15, 10))
    
    # 颜色列表
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 绘制每条预测轨迹
    for i, trajectory in enumerate(trajectories):
        color = colors[i % len(colors)]
        plt.plot(trajectory[:, 0], trajectory[:, 1], 
                color=color, linewidth=2, alpha=0.7,
                label=f'Predicted Trajectory {i+1}', 
                marker='o', markersize=2)
        
        # 添加箭头显示方向 (只在轨迹的几个关键点)
        for j in range(0, len(trajectory)-1, 8):  # 每8个点画一个箭头
            dx = trajectory[j+1, 0] - trajectory[j, 0]
            dy = trajectory[j+1, 1] - trajectory[j, 1]
            plt.arrow(trajectory[j, 0], trajectory[j, 1], dx, dy, 
                     head_width=0.3, head_length=0.2, fc=color, ec=color, alpha=0.6)
    
    # 绘制真实轨迹 (如果可用)
    if ground_truth is not None and len(ground_truth) > 0:
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'k--', 
                linewidth=3, label='Ground Truth', marker='s', markersize=3)
    
    # 标记起始点
    if trajectories:
        plt.plot(trajectories[0][0, 0], trajectories[0][0, 1], 'go', 
                markersize=12, label='Start Point', markeredgewidth=2, markeredgecolor='black')
        
        # 标记各轨迹的结束点
        for i, trajectory in enumerate(trajectories):
            color = colors[i % len(colors)]
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'o', 
                    color=color, markersize=8, markeredgewidth=2, markeredgecolor='black')
    
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.title(f'Multiple Vehicle Trajectory Predictions with Sampler Actions\n'
              f'Start: Row {start_info["index"]}, Time: {start_info["time"]}, '
              f'{len(trajectories)} trajectories', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"多轨迹图保存至: {save_path}")

def main_multiple_trajectories():
    """主函数 - 生成多条轨迹"""
    # 文件路径
    csv_file_path = '/home/jamesye/semester_arbeit/FDM/data/FDM_dataset/scene-0069_merged_data.csv'
    
    # 指定要测试的起始索引和轨迹数量
    specified_start_idx = 29
    n_trajectories = 15  # 生成5条不同的轨迹
    
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        print(f"错误: CSV文件未找到 {csv_file_path}")
        return
    
    try:
        print("=== 加载模型和标准化器 ===")
        # 加载sampler模型和标准化器
        sampler, waypoint_scaler, action_scaler, sampler_device = load_sampler_and_scalers()
        print("Sampler模型加载成功")
        
        # 加载FDM模型和特征标准化器
        fdm_model, feature_scaler, fdm_device = load_fdm_model_and_scaler()
        print("FDM模型加载成功")
        
        print(f"\n=== 处理CSV数据 ===")
        # 预处理CSV数据
        df = preprocess_csv_data(csv_file_path)
        print(f"CSV数据加载成功，共 {len(df)} 行")
        
        print(f"\n=== 从索引 {specified_start_idx} 提取waypoints ===")
        # 提取waypoints
        waypoints, current_state = extract_waypoints_from_csv(df, specified_start_idx)
        
        print(f"\n=== 使用sampler生成 {n_trajectories} 条actions轨迹 ===")
        # 使用sampler生成多条actions
        multiple_sampler_actions = generate_multiple_actions_with_sampler(
            sampler, waypoints, waypoint_scaler, action_scaler, sampler_device, n_samples=n_trajectories)
        
        print(f"\n=== 使用多条sampler actions进行轨迹预测 ===")
        # 使用多条sampler actions和FDM模型进行轨迹预测
        trajectories, ground_truth, start_info, all_states, ground_truth_states = predict_multiple_trajectories_with_sampler_actions(
            fdm_model, df, multiple_sampler_actions, specified_start_idx, feature_scaler)
        
        # 绘制多条轨迹
        save_path = f'multiple_sampler_trajectories_start_idx_{specified_start_idx}_{n_trajectories}trajs.png'
        plot_multiple_trajectories(trajectories, ground_truth, start_info, save_path)
        
        # 计算轨迹统计信息
        print(f"\n=== 多轨迹统计信息 ===")
        print(f"生成的轨迹数量: {len(trajectories)}")
        print(f"起始位置: ({start_info['pos_x']:.2f}, {start_info['pos_y']:.2f})")
        print(f"初始速度: {start_info['vel_x']:.2f} m/s")
        
        for i, trajectory in enumerate(trajectories):
            total_distance = 0
            for j in range(1, len(trajectory)):
                dist = np.sqrt((trajectory[j, 0] - trajectory[j-1, 0])**2 + 
                              (trajectory[j, 1] - trajectory[j-1, 1])**2)
                total_distance += dist
            
            print(f"轨迹 {i+1}: 长度={total_distance:.2f}m, 终点=({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
            
            # 计算与真实轨迹的误差
            if ground_truth is not None:
                final_error = np.sqrt((trajectory[-1, 0] - ground_truth[-1, 0])**2 + 
                                    (trajectory[-1, 1] - ground_truth[-1, 1])**2)
                print(f"        最终位置误差: {final_error:.2f}m")
        
        # 计算轨迹间的多样性
        if len(trajectories) > 1:
            print(f"\n=== 轨迹多样性分析 ===")
            final_positions = np.array([traj[-1] for traj in trajectories])
            distances_between_endpoints = []
            for i in range(len(final_positions)):
                for j in range(i+1, len(final_positions)):
                    dist = np.sqrt(np.sum((final_positions[i] - final_positions[j])**2))
                    distances_between_endpoints.append(dist)
            
            avg_diversity = np.mean(distances_between_endpoints)
            max_diversity = np.max(distances_between_endpoints)
            print(f"终点间平均距离: {avg_diversity:.2f}m")
            print(f"终点间最大距离: {max_diversity:.2f}m")
        
        print(f"\n=== 完成! ===")
        print(f"成功生成并预测了 {len(trajectories)} 条不同的轨迹")
        
    except Exception as e:
        print(f"程序执行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main_multiple_trajectories()