import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import pickle
import argparse
from model import FDMNet, FDMNetPure

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configurations
MODEL_CONFIGS = {
    'fdmnet': {
        'model_class': FDMNet,
        'model_params': {
            'input_size': 6,
            'hidden_size': 125,
            'num_layers': 3,
            'output_size': 18,
            'sequence_length': 5
        },
        'model_path': 'best_model.pth',
        'scaler_path': 'feature_scaler.pkl',
        'output_prefix': ''
    },
    'fdmnetpure': {
        'model_class': FDMNetPure,
        'model_params': {
            'input_size': 6,
            'hidden_size': 25,
            'num_layers': 3,
            'output_size': 3,
            'sequence_length': 5
        },
        'model_path': 'best_model_pure.pth',
        'scaler_path': 'feature_scaler_pure.pkl',
        'output_prefix': 'pure_'
    }
}

def load_feature_scaler(scaler_path='feature_scaler.pkl'):
    """Load the feature scaler used during training"""
    try:
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        print(f"Feature scaler loaded successfully from {scaler_path}")
        return feature_scaler
    except FileNotFoundError:
        print(f"Warning: Feature scaler file not found at {scaler_path}")
        print("Features will NOT be normalized - this may affect prediction accuracy")
        return None

def apply_feature_normalization(features, scaler):
    """Apply feature normalization using the loaded scaler"""
    if scaler is None:
        return features
    
    original_shape = features.shape
    
    if len(original_shape) == 2:
        features_reshaped = features.reshape(-1, original_shape[-1])
        features_normalized = scaler.transform(features_reshaped)
        features_normalized = features_normalized.reshape(original_shape)
    else:
        features_normalized = scaler.transform(features.reshape(1, -1)).flatten()
    
    return features_normalized

def load_trained_model(model_config, model_path=None):
    """
    Load the trained model with specified configuration
    
    Args:
        model_config: Dictionary containing model_class and model_params
        model_path: Path to model checkpoint (optional, uses config default if None)
    
    Returns:
        Loaded model in eval mode
    """
    if model_path is None:
        model_path = model_config['model_path']
    
    # Create model instance
    model_class = model_config['model_class']
    model_params = model_config['model_params']
    
    model = model_class(**model_params).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model {model_class.__name__} loaded successfully from {model_path}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model

def load_evaluation_data(npz_file='../data/evaluation_segments.npz'):
    """Load evaluation data from NPZ file"""
    data = np.load(npz_file, allow_pickle=True)
    
    return {
        'history_features': data['history_features'],
        'initial_states': data['initial_states'],
        'actions_sequence': data['actions_sequence'],
        'ground_truth_trajectories': data['ground_truth_trajectories'],
        'scene_info': data['scene_info']
    }

def predict_trajectory_with_real_actions(model, history_features, initial_state, 
                                         actions_sequence, feature_scaler=None):
    """
    Predict trajectory using real actions
    
    Args:
        model: Trained FDM model
        history_features: (5, 6) historical features
        initial_state: (6,) initial state [pos_x, pos_y, heading, vel_x, vel_y, rotation_rate]
        actions_sequence: (20, 3) real actions sequence [pedal, steer, brake]
        feature_scaler: Feature scaler for normalization
    
    Returns:
        predicted_trajectory: (21, 2) predicted positions [x, y] (including initial position)
        predicted_states: (21, 6) predicted states (including initial state)
    """
    # Apply feature normalization to history
    if feature_scaler is not None:
        history_features = apply_feature_normalization(history_features.copy(), feature_scaler)
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(history_features).unsqueeze(0).to(device)  # (1, 5, 6)
    
    # Extract initial state components
    current_pos_x = initial_state[0]
    current_pos_y = initial_state[1]
    current_heading = initial_state[2]
    current_vel_x = initial_state[3]
    current_vel_y = initial_state[4]
    current_rotation_rate = initial_state[5]
    
    # Store predicted trajectory and states
    predicted_trajectory = [[current_pos_x, current_pos_y]]
    predicted_states = [[current_pos_x, current_pos_y, current_heading, 
                        current_vel_x, current_vel_y, current_rotation_rate]]
    
    # Predict for 20 steps
    with torch.no_grad():
        for step in range(len(actions_sequence)):
            # Get current actions (from real data)
            actions = actions_sequence[step]  # [pedal, steer, brake]
            
            # Create current state tensor for model input
            # Model expects: [pos_x, pos_y, heading, vel_x, vel_y, rotation_rate, pedal, steer, brake]
            current_state_tensor = torch.FloatTensor([
                current_pos_x, current_pos_y, current_heading,
                current_vel_x, current_vel_y, current_rotation_rate,
                actions[0], actions[1], actions[2]
            ]).unsqueeze(0).to(device)
            
            # Model predicts next velocity states
            pred_states = model(features_tensor, current_state_tensor)  # (1, 3)
            
            # Extract predicted states
            pred_vel_x = pred_states[0, 0].item()
            pred_vel_y = 0  # pred_states[0, 1].item()
            pred_rotation_rate = pred_states[0, 2].item()
            
            # Integrate to get position and heading (dt = 0.1s)
            dt = 0.1
            next_pos_x = current_pos_x + dt * (current_vel_x * np.cos(current_heading) - 
                                                current_vel_y * np.sin(current_heading))
            next_pos_y = current_pos_y + dt * (current_vel_x * np.sin(current_heading) + 
                                                current_vel_y * np.cos(current_heading))
            next_heading = current_heading + dt * current_rotation_rate
            
            # Update states
            current_pos_x = next_pos_x
            current_pos_y = next_pos_y
            current_heading = next_heading
            current_vel_x = pred_vel_x
            current_vel_y = pred_vel_y
            current_rotation_rate = pred_rotation_rate
            
            # Store predictions
            predicted_trajectory.append([next_pos_x, next_pos_y])
            predicted_states.append([next_pos_x, next_pos_y, next_heading,
                                    pred_vel_x, pred_vel_y, pred_rotation_rate])
            
            # Update input sequence - sliding window
            new_feature = np.array([pred_vel_x, pred_vel_y, pred_rotation_rate,
                                   actions[0], actions[1], actions[2]], dtype=np.float32)
            
            # Apply normalization to new feature
            if feature_scaler is not None:
                new_feature = apply_feature_normalization(new_feature, feature_scaler)
            
            # Update feature sequence
            features_tensor = torch.cat([
                features_tensor[:, 1:, :],
                torch.FloatTensor(new_feature).unsqueeze(0).unsqueeze(0).to(device)
            ], dim=1)
    
    return np.array(predicted_trajectory), np.array(predicted_states)

def calculate_trajectory_errors(predicted_trajectory, ground_truth_trajectory):
    """
    Calculate final position error between predicted and ground truth endpoint
    
    Args:
        predicted_trajectory: (21, 2) predicted [x, y] positions
        ground_truth_trajectory: (20, 6) ground truth states
    
    Returns:
        final_error: Final position error (distance at endpoint)
    """
    # Extract ground truth final position
    gt_final_position = ground_truth_trajectory[-1, :2]  # Final [pos_x, pos_y]
    
    # Get predicted final position (last position)
    pred_final_position = predicted_trajectory[-1, :]  # (2,) [x, y]
    
    # Calculate Euclidean distance between final positions
    final_error = np.sqrt(np.sum((pred_final_position - gt_final_position)**2))
    
    return final_error

def evaluate_all_segments(model, eval_data, feature_scaler, num_samples=None, verbose=False):
    """
    Evaluate model on all segments from evaluation data
    
    Args:
        model: Trained FDM model
        eval_data: Dictionary containing evaluation segments
        feature_scaler: Feature scaler
        num_samples: Number of samples to evaluate (None for all)
        verbose: Print detailed information for each sample
    
    Returns:
        results: Dictionary containing aggregated results
    """
    history_features = eval_data['history_features']
    initial_states = eval_data['initial_states']
    actions_sequence = eval_data['actions_sequence']
    ground_truth_trajectories = eval_data['ground_truth_trajectories']
    
    # Determine number of samples to evaluate
    total_samples = len(history_features)
    if num_samples is None or num_samples > total_samples:
        num_samples = total_samples
    
    print(f"\nEvaluating {num_samples} segments...")
    
    # Store all final errors
    all_final_errors = []
    
    for i in range(num_samples):
        # Predict trajectory using real actions
        pred_trajectory, pred_states = predict_trajectory_with_real_actions(
            model,
            history_features[i],
            initial_states[i],
            actions_sequence[i],
            feature_scaler
        )
        
        # Calculate final position error
        final_error = calculate_trajectory_errors(pred_trajectory, ground_truth_trajectories[i])
        
        all_final_errors.append(final_error)
        
        if verbose and i < 5:  # Print details for first 5 samples
            print(f"\nSample {i}:")
            print(f"  Final position error: {final_error:.4f} m")
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_samples} segments...")
    
    # Filter out top 0.1% errors
    all_final_errors_array = np.array(all_final_errors)
    threshold_percentile = 99.9  # Keep bottom 99.9%, remove top 0.1%
    error_threshold = np.percentile(all_final_errors_array, threshold_percentile)
    
    # Filter errors
    filtered_errors = all_final_errors_array[all_final_errors_array <= error_threshold]
    num_removed = len(all_final_errors_array) - len(filtered_errors)
    
    print(f"\nFiltering: Removed top 0.1% ({num_removed} samples) with errors > {error_threshold:.4f}m")
    
    # Find top 5 worst samples (before filtering)
    sorted_indices = np.argsort(all_final_errors_array)[::-1]  # Sort descending
    worst_5_indices = sorted_indices[:5]
    worst_5_errors = all_final_errors_array[worst_5_indices]

    # Aggregate results
    results = {
        'num_samples': num_samples,
        'num_filtered': len(filtered_errors),
        'num_removed': num_removed,
        'error_threshold': error_threshold,
        'mean_final_error': np.mean(filtered_errors),
        'std_final_error': np.std(filtered_errors),
        'min_final_error': np.min(filtered_errors),
        'max_final_error': np.max(filtered_errors),
        'all_final_errors': filtered_errors.tolist(),
        'all_final_errors_unfiltered': all_final_errors,
        'worst_5_indices': worst_5_indices.tolist(),
        'worst_5_errors': worst_5_errors.tolist()
    }
    
    return results

def plot_error_distribution(results, model_name='FDMNet', save_path='error_distribution.png'):
    """Plot final position error distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Final position error distribution
    ax.hist(results['all_final_errors'], bins=50, edgecolor='black', color='skyblue')
    ax.set_xlabel('Final Position Error (m)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{model_name} - Final Position Error Distribution (at 2s prediction horizon)\n'
                 f'Mean: {results["mean_final_error"]:.4f}m ± {results["std_final_error"]:.4f}m\n'
                 f'Min: {results["min_final_error"]:.4f}m, Max: {results["max_final_error"]:.4f}m\n'
                 f'(Top 0.1% excluded: {results["num_removed"]} samples, threshold: {results["error_threshold"]:.4f}m)',
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axvline(results['mean_final_error'], color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {results["mean_final_error"]:.4f}m')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Error distribution plot saved to: {save_path}")

def plot_sample_trajectories(model, eval_data, feature_scaler, num_samples=5, save_path='sample_trajectories.png'):
    """Plot sample trajectory predictions"""
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Predict trajectory
        pred_trajectory, _ = predict_trajectory_with_real_actions(
            model,
            eval_data['history_features'][i],
            eval_data['initial_states'][i],
            eval_data['actions_sequence'][i],
            feature_scaler
        )
        
        # Get ground truth
        gt_trajectory = eval_data['ground_truth_trajectories'][i]
        gt_positions = gt_trajectory[:, :2]      

        # Calculate error
        errors = calculate_trajectory_errors(pred_trajectory, gt_trajectory)
        
        # Plot
        axes[i].plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'b-', 
                    linewidth=2, marker='o', markersize=3, label='Predicted')
        axes[i].plot(gt_positions[:, 0], gt_positions[:, 1], 'r--',
                    linewidth=2, marker='s', markersize=3, label='Ground Truth')
        axes[i].plot(pred_trajectory[0, 0], pred_trajectory[0, 1], 'go', 
                    markersize=10, label='Start')
        
        axes[i].set_xlabel('X Position (m)')
        axes[i].set_ylabel('Y Position (m)')
        axes[i].set_title(f'Sample {i}\nFinal Position Error: {errors:.3f}m')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        axes[i].axis('equal')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Sample trajectories plot saved to: {save_path}")

def plot_worst_trajectories(model, eval_data, feature_scaler, worst_indices, worst_errors, 
                            save_path='worst_trajectories.png'):
    """Plot trajectories with the highest errors"""
    num_samples = len(worst_indices)
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (sample_idx, error) in enumerate(zip(worst_indices, worst_errors)):
        # Predict trajectory
        pred_trajectory, _ = predict_trajectory_with_real_actions(
            model,
            eval_data['history_features'][sample_idx],
            eval_data['initial_states'][sample_idx],
            eval_data['actions_sequence'][sample_idx],
            feature_scaler
        )
        
        # Get ground truth
        gt_trajectory = eval_data['ground_truth_trajectories'][sample_idx]
        gt_positions = gt_trajectory[:, :2]
        
        # Get initial position for complete ground truth
        initial_pos = eval_data['initial_states'][sample_idx][:2]
        gt_positions_full = np.vstack([initial_pos, gt_positions])
        
        # Plot
        axes[idx].plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'b-', 
                      linewidth=2, marker='o', markersize=3, label='Predicted')
        axes[idx].plot(gt_positions_full[:, 0], gt_positions_full[:, 1], 'r--',
                      linewidth=2, marker='s', markersize=3, label='Ground Truth')
        axes[idx].plot(pred_trajectory[0, 0], pred_trajectory[0, 1], 'go', 
                      markersize=10, label='Start')
        axes[idx].plot(pred_trajectory[-1, 0], pred_trajectory[-1, 1], 'bs', 
                      markersize=10, label='Pred End')
        axes[idx].plot(gt_positions_full[-1, 0], gt_positions_full[-1, 1], 'rs', 
                      markersize=10, label='GT End')
        
        axes[idx].set_xlabel('X Position (m)', fontsize=11)
        axes[idx].set_ylabel('Y Position (m)', fontsize=11)
        axes[idx].set_title(f'Sample #{sample_idx}\nError: {error:.3f}m', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axis('equal')
    
    plt.suptitle('Top 5 Worst Prediction Samples', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Worst trajectories plot saved to: {save_path}")

def main(model_type='fdmnet'):
    """
    Main function
    
    Args:
        model_type: Type of model to evaluate ('fdmnet' or 'fdmnetpure')
    """
    # Get model configuration
    if model_type not in MODEL_CONFIGS:
        print(f"Error: Unknown model type '{model_type}'")
        print(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return
    
    model_config = MODEL_CONFIGS[model_type]
    output_prefix = model_config['output_prefix']
    
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_config['model_class'].__name__}")
    print(f"{'='*60}\n")
    
    # File paths
    model_path = model_config['model_path']
    eval_data_path = '../data/evaluation_segments.npz'
    scaler_path = model_config['scaler_path']
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found {model_path}")
        return
    
    if not os.path.exists(eval_data_path):
        print(f"Error: Evaluation data file not found {eval_data_path}")
        print("Please run data/data_evaluate_FDM.py first to generate evaluation data")
        return
    
    try:
        # Load feature scaler
        feature_scaler = load_feature_scaler(scaler_path)
        
        # Load model
        model = load_trained_model(model_config, model_path)
        
        # Load evaluation data
        print("\nLoading evaluation data...")
        eval_data = load_evaluation_data(eval_data_path)
        print(f"Loaded {len(eval_data['history_features'])} evaluation segments")
        
        # Evaluate on all segments (or specify num_samples for subset)
        results = evaluate_all_segments(
            model, 
            eval_data, 
            feature_scaler, 
            num_samples=None,  # Set to specific number for quick test, None for all
            verbose=True
        )
        
        # Print summary statistics
        print("\n" + "="*60)
        print("EVALUATION RESULTS SUMMARY")
        print("="*60)
        print(f"Model: {model_config['model_class'].__name__}")
        print(f"Number of samples evaluated: {results['num_samples']}")
        print(f"Number of samples after filtering: {results['num_filtered']}")
        print(f"Number of samples removed (top 0.1%): {results['num_removed']}")
        print(f"Error threshold (99.9 percentile): {results['error_threshold']:.4f} m")
        print(f"\nFinal Position Error (at 2s):")
        print(f"  Mean: {results['mean_final_error']:.4f} m")
        print(f"  Std:  {results['std_final_error']:.4f} m")
        print(f"  Min:  {results['min_final_error']:.4f} m")
        print(f"  Max:  {results['max_final_error']:.4f} m")
        print("="*60)
        
        # Plot error distribution
        plot_error_distribution(
            results, 
            model_name=model_config['model_class'].__name__,
            save_path=f'{output_prefix}error_distribution.png'
        )
        
        # Plot sample trajectories
        plot_sample_trajectories(
            model, eval_data, feature_scaler, 
            num_samples=5, 
            save_path=f'{output_prefix}sample_trajectories.png'
        )
        
        # Plot worst 5 trajectories
        print("\n" + "="*60)
        print("WORST 5 SAMPLES")
        print("="*60)
        for idx, (sample_idx, error) in enumerate(zip(results['worst_5_indices'], 
                                                       results['worst_5_errors'])):
            print(f"{idx+1}. Sample #{sample_idx}: Error = {error:.4f}m")
        print("="*60 + "\n")
        
        plot_worst_trajectories(
            model, eval_data, feature_scaler,
            results['worst_5_indices'],
            results['worst_5_errors'],
            save_path=f'{output_prefix}worst_trajectories.png'
        )
                               
    except Exception as e:
        print(f"Program execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate FDM models on trajectory prediction')
    parser.add_argument('--model', type=str, default='fdmnet', 
                       choices=['fdmnet', 'fdmnetpure'],
                       help='Model type to evaluate (default: fdmnet)')
    
    args = parser.parse_args()
    
    main(model_type=args.model)