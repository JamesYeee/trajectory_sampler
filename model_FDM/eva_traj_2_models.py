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
            'hidden_size': 125,
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

def predict_trajectory_batch(model, history_features_batch, initial_states_batch, 
                             actions_sequence_batch, feature_scaler=None, batch_size=32):
    """
    批量预测轨迹（支持并行计算）
    
    Args:
        model: Trained FDM model
        history_features_batch: (B, 5, 6) batch of historical features
        initial_states_batch: (B, 6) batch of initial states
        actions_sequence_batch: (B, 20, 3) batch of actions sequences
        feature_scaler: Feature scaler for normalization
        batch_size: Batch size for inference
    
    Returns:
        predicted_trajectories: (B, 21, 2) predicted positions
        predicted_states: (B, 21, 6) predicted states
    """
    B = len(history_features_batch)
    
    # Apply feature normalization to history
    if feature_scaler is not None:
        history_features_batch = np.array([
            apply_feature_normalization(hf.copy(), feature_scaler) 
            for hf in history_features_batch
        ])
    
    # Convert to tensors
    features_tensor = torch.FloatTensor(history_features_batch).to(device)  # (B, 5, 6)
    initial_states_tensor = torch.FloatTensor(initial_states_batch).to(device)  # (B, 6)
    
    # Initialize current states
    current_pos_x = initial_states_tensor[:, 0].clone()
    current_pos_y = initial_states_tensor[:, 1].clone()
    current_heading = initial_states_tensor[:, 2].clone()
    current_vel_x = initial_states_tensor[:, 3].clone()
    current_vel_y = initial_states_tensor[:, 4].clone()
    current_rotation_rate = initial_states_tensor[:, 5].clone()
    
    # Store trajectories
    predicted_trajectories = torch.stack([current_pos_x, current_pos_y], dim=1).unsqueeze(1)  # (B, 1, 2)
    predicted_states_list = [torch.stack([
        current_pos_x, current_pos_y, current_heading,
        current_vel_x, current_vel_y, current_rotation_rate
    ], dim=1)]  # List of (B, 6)
    
    # Predict for 20 steps
    with torch.no_grad():
        for step in range(20):
            # Get current actions for all samples in batch
            actions = torch.FloatTensor(actions_sequence_batch[:, step, :]).to(device)  # (B, 3)
            
            # Create current state tensor for model input
            current_state_tensor = torch.cat([
                current_pos_x.unsqueeze(1),
                current_pos_y.unsqueeze(1),
                current_heading.unsqueeze(1),
                current_vel_x.unsqueeze(1),
                current_vel_y.unsqueeze(1),
                current_rotation_rate.unsqueeze(1),
                actions
            ], dim=1)  # (B, 9)
            
            # Model predicts next velocity states (batch)
            pred_states = model(features_tensor, current_state_tensor)  # (B, 3)
            
            # Extract predicted states
            pred_vel_x = pred_states[:, 0]
            pred_vel_y = torch.zeros_like(pred_vel_x)  # pred_states[:, 1]
            pred_rotation_rate = pred_states[:, 2]
            
            # Integrate to get position and heading (dt = 0.1s)
            dt = 0.1
            next_pos_x = current_pos_x + dt * (current_vel_x * torch.cos(current_heading) - 
                                                current_vel_y * torch.sin(current_heading))
            next_pos_y = current_pos_y + dt * (current_vel_x * torch.sin(current_heading) + 
                                                current_vel_y * torch.cos(current_heading))
            next_heading = current_heading + dt * current_rotation_rate
            
            # Update states
            current_pos_x = next_pos_x
            current_pos_y = next_pos_y
            current_heading = next_heading
            current_vel_x = pred_vel_x
            current_vel_y = pred_vel_y
            current_rotation_rate = pred_rotation_rate
            
            # Store predictions
            predicted_trajectories = torch.cat([
                predicted_trajectories,
                torch.stack([next_pos_x, next_pos_y], dim=1).unsqueeze(1)
            ], dim=1)
            
            predicted_states_list.append(torch.stack([
                next_pos_x, next_pos_y, next_heading,
                pred_vel_x, pred_vel_y, pred_rotation_rate
            ], dim=1))
            
            # Update input sequence - sliding window
            new_features = torch.stack([
                pred_vel_x, pred_vel_y, pred_rotation_rate,
                actions[:, 0], actions[:, 1], actions[:, 2]
            ], dim=1)  # (B, 6)
            
            # Apply normalization to new features
            if feature_scaler is not None:
                new_features_np = new_features.cpu().numpy()
                new_features_normalized = np.array([
                    apply_feature_normalization(nf, feature_scaler) 
                    for nf in new_features_np
                ])
                new_features = torch.FloatTensor(new_features_normalized).to(device)
            
            # Update feature sequence
            features_tensor = torch.cat([
                features_tensor[:, 1:, :],
                new_features.unsqueeze(1)
            ], dim=1)
    
    predicted_trajectories = predicted_trajectories.cpu().numpy()  # (B, 21, 2)
    predicted_states = torch.stack(predicted_states_list, dim=1).cpu().numpy()  # (B, 21, 6)
    
    return predicted_trajectories, predicted_states

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

def evaluate_all_segments_fast(model, eval_data, feature_scaler, batch_size=64, num_samples=None, verbose=False):
    """
    快速批量评估所有segments（并行计算版本）
    
    Args:
        model: Trained FDM model
        eval_data: Dictionary containing evaluation segments
        feature_scaler: Feature scaler
        batch_size: Batch size for parallel inference
        num_samples: Number of samples to evaluate (None for all)
        verbose: Print detailed information
    
    Returns:
        results: Dictionary containing aggregated results
    """
    history_features = eval_data['history_features']
    initial_states = eval_data['initial_states']
    actions_sequence = eval_data['actions_sequence']
    ground_truth_trajectories = eval_data['ground_truth_trajectories']
    
    # Determine number of samples
    total_samples = len(history_features)
    if num_samples is None or num_samples > total_samples:
        num_samples = total_samples
    
    print(f"\nEvaluating {num_samples} segments with batch size {batch_size}...")
    
    all_final_errors = []
    
    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx
        
        # Prepare batch data
        batch_history = history_features[start_idx:end_idx]
        batch_initial = initial_states[start_idx:end_idx]
        batch_actions = actions_sequence[start_idx:end_idx]
        batch_gt = ground_truth_trajectories[start_idx:end_idx]
        
        # Batch prediction
        pred_trajectories, _ = predict_trajectory_batch(
            model, batch_history, batch_initial, batch_actions, feature_scaler, batch_size
        )
        
        # Calculate errors for this batch
        for i in range(current_batch_size):
            gt_final_position = batch_gt[i][-1, :2]
            pred_final_position = pred_trajectories[i][-1, :]
            final_error = np.sqrt(np.sum((pred_final_position - gt_final_position)**2))
            all_final_errors.append(final_error)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"Processed {end_idx}/{num_samples} segments...")
    
    # Filter out top 0.1% errors
    all_final_errors_array = np.array(all_final_errors)
    threshold_percentile = 99.9
    error_threshold = np.percentile(all_final_errors_array, threshold_percentile)
    
    filtered_errors = all_final_errors_array[all_final_errors_array <= error_threshold]
    num_removed = len(all_final_errors_array) - len(filtered_errors)
    
    print(f"\nFiltering: Removed top 0.1% ({num_removed} samples) with errors > {error_threshold:.4f}m")
    
    # Find top 5 worst samples
    sorted_indices = np.argsort(all_final_errors_array)[::-1]
    worst_5_indices = sorted_indices[:5]
    worst_5_errors = all_final_errors_array[worst_5_indices]
    
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
    # Randomly select sample indices
    total_samples = len(eval_data['history_features'])
    random_indices = random.sample(range(total_samples), num_samples)
    
    # Gather data for randomly selected samples
    selected_history = np.array([eval_data['history_features'][idx] for idx in random_indices])
    selected_initial = np.array([eval_data['initial_states'][idx] for idx in random_indices])
    selected_actions = np.array([eval_data['actions_sequence'][idx] for idx in random_indices])
    
    # Batch predict all samples at once
    pred_trajectories, _ = predict_trajectory_batch(
        model,
        selected_history,
        selected_initial,
        selected_actions,
        feature_scaler
    )
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        sample_idx = random_indices[i]  # Get actual sample index
        pred_trajectory = pred_trajectories[i]  # (21, 2)
        
        # Get ground truth
        gt_trajectory = eval_data['ground_truth_trajectories'][sample_idx]
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
        axes[i].set_title(f'Sample #{sample_idx}\nFinal Position Error: {errors:.3f}m')
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
    
    # Gather data for all worst samples
    worst_history = np.array([eval_data['history_features'][idx] for idx in worst_indices])
    worst_initial = np.array([eval_data['initial_states'][idx] for idx in worst_indices])
    worst_actions = np.array([eval_data['actions_sequence'][idx] for idx in worst_indices])
    
    # Batch predict all worst samples at once
    pred_trajectories, _ = predict_trajectory_batch(
        model,
        worst_history,
        worst_initial,
        worst_actions,
        feature_scaler
    )
    
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]
    
    for idx, (sample_idx, error) in enumerate(zip(worst_indices, worst_errors)):
        pred_trajectory = pred_trajectories[idx]  # (21, 2)
        
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

    # Create results folder (if not exists)
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)

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
        results = evaluate_all_segments_fast(
            model, 
            eval_data, 
            feature_scaler, 
            batch_size=128,  
            num_samples=None,
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
            save_path=f'{results_dir}/{output_prefix}error_distribution.png'
        )
        
        # Plot sample trajectories
        plot_sample_trajectories(
            model, eval_data, feature_scaler, 
            num_samples=5, 
            save_path=f'{results_dir}/{output_prefix}sample_trajectories.png'
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
            save_path=f'{results_dir}/{output_prefix}worst_trajectories.png'
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


# python eva_traj_2_models.py --model fdmnet
# python eva_traj_2_models.py --model fdmnetpure