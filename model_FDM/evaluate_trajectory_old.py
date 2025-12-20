import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle
from model import FDMNet

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_feature_scaler(scaler_path='feature_scaler.pkl'):
    """Load the feature scaler used during training"""
    try:
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print("Feature scaler loaded successfully")
        return feature_scaler
    except FileNotFoundError:
        print(f"Warning: Feature scaler file not found at {scaler_path}")
        print("Features will NOT be normalized - this may affect prediction accuracy")
        return None

def apply_feature_normalization(features, scaler):
    """Apply feature normalization using the loaded scaler"""
    if scaler is None:
        return features
    
    # features shape: (sequence_length, num_features) or (num_features,)
    original_shape = features.shape
    
    if len(original_shape) == 2:
        # For sequence data: (sequence_length, num_features)
        features_reshaped = features.reshape(-1, original_shape[-1])
        features_normalized = scaler.transform(features_reshaped)
        features_normalized = features_normalized.reshape(original_shape)
    else:
        # For single feature vector: (num_features,)
        features_normalized = scaler.transform(features.reshape(1, -1)).flatten()
    
    return features_normalized

def load_trained_model(model_path='best_model.pth'):
    """Load the trained model"""
    model = FDMNet(
        input_size=6,
        hidden_size=125,
        num_layers=3,
        output_size=18,
        sequence_length=5
    ).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully, validation loss: {checkpoint['val_loss']:.6f}")
    return model

def preprocess_csv_data(csv_file_path):
    """Preprocess CSV data, consistent with training"""
    df = pd.read_csv(csv_file_path)
    
    # Data preprocessing - consistent with training
    min_brake = df['brake_sensor'].min()
    df['brake_sensor_scaled'] = (df['brake_sensor'] - min_brake) / (0.631 - min_brake)
    df['pedal_cc_scaled'] = df['pedal_cc'] / 1000
    
    return df

def create_sequence_from_start(df, start_idx, sequence_length=5, feature_scaler=None):
    """Create input sequence from starting point with feature normalization"""
    if start_idx + sequence_length > len(df):
        return None, None
    
    # Feature columns - consistent with training
    feature_columns = ['vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    current_columns = ['pos_x', 'pos_y', 'heading', 'vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    
    # Extract sequence features (t-4 to t)
    sequence_data = df.iloc[start_idx:start_idx + sequence_length]
    features = sequence_data[feature_columns].values.astype(np.float32)
    
    # Apply feature normalization
    if feature_scaler is not None:
        features = apply_feature_normalization(features, feature_scaler)
        print("Applied feature normalization to input sequence")
    
    # Current state (at time t) - poses are not normalized
    current_state = df.iloc[start_idx + sequence_length - 1][current_columns].values.astype(np.float32)
    
    return features, current_state

def predict_trajectory(model, csv_file_path, start_idx=None, prediction_steps=20, feature_scaler=None):
    """
    Predict trajectory with feature normalization
    
    Args:
        model: Trained model
        csv_file_path: Path to CSV file
        start_idx: Starting index, randomly selected if None
        prediction_steps: Number of prediction steps (20 steps = 2 seconds, since dt=0.1s)
        feature_scaler: Feature scaler for normalization
    
    Returns:
        trajectory: Predicted trajectory
        ground_truth: Ground truth trajectory (if available)
        start_info: Starting point information
    """
    # Load and preprocess data
    df = preprocess_csv_data(csv_file_path)
    
    # Randomly select starting point
    if start_idx is None:
        # Ensure sufficient data for sequence and prediction
        max_start = len(df) - 5 - prediction_steps
        if max_start <= 5:
            raise ValueError("Insufficient data in CSV file for prediction")
        start_idx = random.randint(5, max_start)
    
    print(f"Selected starting point: Row {start_idx}")
    
    # Create initial sequence with normalization
    features, current_state = create_sequence_from_start(df, start_idx - 5, 5, feature_scaler)
    if features is None:
        raise ValueError("Unable to create valid sequence data")
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)  # (1, 5, 6)
    current_state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(device)  # (1, 9)
    
    # Store predicted trajectory
    trajectory = []
    states = []
    
    # Initial state
    current_pos_x = current_state[0]
    current_pos_y = current_state[1] 
    current_heading = current_state[2]
    current_vel_x = current_state[3]
    current_vel_y = current_state[4]
    current_rotation_rate = current_state[5]
    
    trajectory.append([current_pos_x, current_pos_y])
    states.append([current_vel_x, current_vel_y, current_rotation_rate])
    
    # Predict subsequent trajectory
    with torch.no_grad():
        # Print header for step-by-step output
        print(f"\n--- Step-by-Step Prediction Details ---")
        print("Step | Pred_v_x | Pred_v_y | Pred_rotation | Actions (pedal, steer, brake)")
        print("-" * 65)
        
        for step in range(prediction_steps):
            # Model predicts next step velocity states
            pred_states = model(features_tensor, current_state_tensor)  # (1, 3)
            
            # Extract predicted states
            pred_vel_x = pred_states[0, 0].item()
            pred_vel_y = 0 #pred_states[0, 1].item()
            pred_rotation_rate = pred_states[0, 2].item()
            
            # Get current actions (assume constant or use next value from CSV)
            if start_idx + step < len(df):
                next_actions = df.iloc[start_idx + step][['pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']].values
            else:
                # If beyond CSV range, use last action
                next_actions = df.iloc[-1][['pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']].values
            
            # Print step details
            print(f"{step:4d} | {pred_vel_x:8.4f} | {pred_vel_y:8.4f} | {pred_rotation_rate:13.4f} | ({next_actions[0]:.3f}, {next_actions[1]:.3f}, {next_actions[2]:.3f})")
            
            # Integrate to get position and heading (dt = 0.1s)
            dt = 0.1
            next_pos_x = current_pos_x + dt * (current_vel_x * np.cos(current_heading) - current_vel_y * np.sin(current_heading))
            next_pos_y = current_pos_y + dt * (current_vel_x * np.sin(current_heading) + current_vel_y * np.cos(current_heading))
            next_heading = current_heading + dt * current_rotation_rate
            
            trajectory.append([next_pos_x, next_pos_y])
            states.append([pred_vel_x, pred_vel_y, pred_rotation_rate])

            # Update states
            current_pos_x = next_pos_x
            current_pos_y = next_pos_y
            current_heading = next_heading
            current_vel_x = pred_vel_x
            current_vel_y = pred_vel_y
            current_rotation_rate = pred_rotation_rate
            
            # Update input sequence - sliding window
            # Create new feature vector [vel_x, vel_y, rotation_rate, pedal_cc_scaled, steer_corrected, brake_sensor_scaled]
            new_feature = np.array([pred_vel_x, pred_vel_y, pred_rotation_rate, 
                                  next_actions[0], next_actions[1], next_actions[2]], dtype=np.float32)
            
            # Apply normalization to the new feature vector
            if feature_scaler is not None:
                new_feature = apply_feature_normalization(new_feature, feature_scaler)
            
            # Update feature sequence (sliding window)
            features_tensor = torch.cat([features_tensor[:, 1:, :], 
                                       torch.FloatTensor(new_feature).unsqueeze(0).unsqueeze(0).to(device)], dim=1)
            
            # Update current state tensor (poses are not normalized)
            current_state_tensor = torch.FloatTensor([
                current_pos_x, current_pos_y, current_heading,
                current_vel_x, current_vel_y, current_rotation_rate,
                next_actions[0], next_actions[1], next_actions[2]
            ]).unsqueeze(0).to(device)
    
    # Get ground truth trajectory (if sufficient data in CSV)
    ground_truth = []
    ground_truth_states = []
    if start_idx + prediction_steps < len(df):
        for i in range(prediction_steps + 1):
            gt_pos_x = df.iloc[start_idx + i]['pos_x']
            gt_pos_y = df.iloc[start_idx + i]['pos_y']
            ground_truth.append([gt_pos_x, gt_pos_y])

            # Get ground truth states (vel_x, vel_y, rotation_rate)
            gt_vel_x = df.iloc[start_idx + i]['vel_x']
            gt_vel_y = df.iloc[start_idx + i]['vel_y']
            gt_rotation_rate = df.iloc[start_idx + i]['rotation_rate']
            ground_truth_states.append([gt_vel_x, gt_vel_y, gt_rotation_rate])
    
    # Starting point information
    start_info = {
        'index': start_idx,
        'time': df.iloc[start_idx]['utime'],
        'pos_x': current_state[0],
        'pos_y': current_state[1],
        'vel_x': current_state[3],
        'vel_y': current_state[4]
    }
    
    return (np.array(trajectory), 
            np.array(ground_truth) if ground_truth else None, 
            start_info, 
            np.array(states),
            np.array(ground_truth_states) if ground_truth_states else None)

# ... existing code ...
def plot_trajectory(trajectory, ground_truth=None, start_info=None, save_path='trajectory_evaluation.png'):
    """Plot predicted trajectory"""
    plt.figure(figsize=(12, 8))
    
    # Plot predicted trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, label='Predicted Trajectory', marker='o', markersize=3)
    
    # Plot ground truth trajectory (if available)
    if ground_truth is not None and len(ground_truth) > 0:
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'r--', linewidth=2, label='Ground Truth', marker='s', markersize=3)
    
    # Mark starting point
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='Start Point')
    
    # Mark ending point
    plt.plot(trajectory[-1, 0], trajectory[-1, 1], 'ro', markersize=10, label='End Point')
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title(f'Vehicle Trajectory Prediction (2-second prediction)\nStart: Row {start_info["index"]}, Time: {start_info["time"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Add arrows to show direction
    for i in range(0, len(trajectory)-1, 5):  # Draw arrow every 5 points
        dx = trajectory[i+1, 0] - trajectory[i, 0]
        dy = trajectory[i+1, 1] - trajectory[i, 1]
        plt.arrow(trajectory[i, 0], trajectory[i, 1], dx, dy, 
                 head_width=0.5, head_length=0.3, fc='blue', ec='blue', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Trajectory plot saved to: {save_path}")

def evaluate_multiple_trajectories(model, csv_file_path, feature_scaler, num_evaluations=5):
    """Evaluate trajectory predictions from multiple random starting points"""
    print(f"Evaluating {num_evaluations} random trajectories...")
    
    for i in range(num_evaluations):
        print(f"\n=== Evaluation {i+1}/{num_evaluations} ===")
        
        try:
            trajectory, ground_truth, start_info, states, ground_truth_states = predict_trajectory(
                model, csv_file_path, feature_scaler=feature_scaler)
            
            # Print rotation rate comparison
            print(f"\n--- Rotation Rate Comparison (Case {i+1}) ---")
            print("Step | Predicted | Ground Truth | Error")
            print("-" * 40)
            
            if ground_truth_states is not None:
                # Compare rotation rates for each prediction step
                rotation_rate_errors = []
                for step in range(min(len(states), len(ground_truth_states))):
                    pred_rot = states[step, 2]  # rotation_rate is index 2
                    gt_rot = ground_truth_states[step, 2]
                    error = abs(pred_rot - gt_rot)
                    rotation_rate_errors.append(error)
                    print(f"{step:4d} | {pred_rot:9.4f} | {gt_rot:12.4f} | {error:5.4f}")
                
                # Print rotation rate statistics
                mean_rot_error = np.mean(rotation_rate_errors)
                max_rot_error = np.max(rotation_rate_errors)
                print(f"\nRotation Rate Statistics:")
                print(f"Mean absolute error: {mean_rot_error:.4f} rad/s")
                print(f"Max absolute error: {max_rot_error:.4f} rad/s")
            else:
                print("Ground truth rotation rates not available for comparison")
                print("Predicted rotation rates:")
                for step, state in enumerate(states):
                    print(f"Step {step:2d}: {state[2]:.4f} rad/s")
            
            # Calculate trajectory statistics
            total_distance = 0
            for j in range(1, len(trajectory)):
                dist = np.sqrt((trajectory[j, 0] - trajectory[j-1, 0])**2 + 
                              (trajectory[j, 1] - trajectory[j-1, 1])**2)
                total_distance += dist
            
            print(f"\nTrajectory Statistics:")
            print(f"Predicted trajectory length: {total_distance:.2f} m")
            print(f"Initial velocity: {start_info['vel_x']:.2f} m/s")
            print(f"Final position: ({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
            
            # Calculate error if ground truth is available
            if ground_truth is not None:
                final_error = np.sqrt((trajectory[-1, 0] - ground_truth[-1, 0])**2 + 
                                    (trajectory[-1, 1] - ground_truth[-1, 1])**2)
                print(f"Final position error: {final_error:.2f} m")
            
            # Plot trajectory
            save_path = f'trajectory_evaluation_{i+1}.png'
            plot_trajectory(trajectory, ground_truth, start_info, save_path)
            
        except Exception as e:
            print(f"Error evaluating trajectory {i+1}: {e}")

def main():
    """Main function"""
    # File paths
    model_path = 'best_model.pth'
    csv_file_path = '../data/FDM_dataset/scene-0069_merged_data.csv'
    scaler_path = 'feature_scaler.pkl'
    
    # 指定要测试的起始索引 - 在这里修改你想要的start_idx
    specified_start_idx = 29  # 修改这个值来测试不同的起始点
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        return
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found {csv_file_path}")
        return
    
    try:
        # Load feature scaler
        feature_scaler = load_feature_scaler(scaler_path)
        
        # Load model
        model = load_trained_model(model_path)
        
        # Single prediction with specified start_idx
        print(f"\n=== Trajectory Prediction from Index {specified_start_idx} ===")
        trajectory, ground_truth, start_info, states, ground_truth_states = predict_trajectory(
            model, csv_file_path, start_idx=specified_start_idx, feature_scaler=feature_scaler)
        
        # Print rotation rate comparison for single prediction
        print(f"\n--- Rotation Rate Comparison (Start Index {specified_start_idx}) ---")
        print("Step | Predicted | Ground Truth | Error")
        print("-" * 40)
        
        if ground_truth_states is not None:
            for step in range(min(len(states), len(ground_truth_states))):
                pred_rot = states[step, 2]
                gt_rot = ground_truth_states[step, 2]
                error = abs(pred_rot - gt_rot)
                print(f"{step:4d} | {pred_rot:9.4f} | {gt_rot:12.4f} | {error:5.4f}")
        
        plot_trajectory(trajectory, ground_truth, start_info, f'trajectory_start_idx_{specified_start_idx}.png')
        
        # 如果你想测试多个指定的起始点，可以取消下面的注释
        # specified_indices = [50, 100, 150, 200]  # 指定多个起始点
        # print(f"\n=== Testing Multiple Specified Start Indices: {specified_indices} ===")
        # for i, start_idx in enumerate(specified_indices):
        #     print(f"\n--- Testing Start Index {start_idx} ({i+1}/{len(specified_indices)}) ---")
        #     try:
        #         trajectory, ground_truth, start_info, states, ground_truth_states = predict_trajectory(
        #             model, csv_file_path, start_idx=start_idx, feature_scaler=feature_scaler)
        #         plot_trajectory(trajectory, ground_truth, start_info, f'trajectory_start_idx_{start_idx}.png')
        #         print(f"Successfully predicted trajectory from index {start_idx}")
        #     except Exception as e:
        #         print(f"Error predicting from index {start_idx}: {e}")
        
    except Exception as e:
        print(f"Program execution error: {e}")

if __name__ == "__main__":
    main()