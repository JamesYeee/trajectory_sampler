import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
#brake sensor, find min als 0
#       scale from [min, 0.631] to [0,1]

#use raw brake, steering, throttle[0, 1000] / 1000(also to [0,1])

#Input 5 step v_value, 5 step input value
#Label next step v value,
#then calculate pose

#time_slice structur, feature: [t-4, t] v + u
#                     label: [t+1] v
#                     current pose + action: [t] x, v, u


def to_time_slice(filename):

    df = pd.read_csv(filename)

    window_size = 6
    step_size = 1
    min_brake = df['brake_sensor'].min()
    df['brake_sensor_scaled'] = (df['brake_sensor'] - min_brake) / (0.631 - min_brake)
    df['pedal_cc_scaled'] = df['pedal_cc'] / 1000
    
    feature_columns = ['vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    label_columns = ['vel_x', 'vel_y', 'rotation_rate']
    current_columns = ['pos_x', 'pos_y', 'heading', 'vel_x', 'vel_y', 'rotation_rate', 'pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']

    features_list = []
    label_list = []
    current_list = []
    
    num_windows = (len(df) - window_size) // step_size +1
    for i in range(num_windows):
        
        time_slice = df.iloc[i:i+window_size]
        
        feature_slice = time_slice.iloc[:5][feature_columns]
        
        # 检查feature中的vel_x是否都大于等于0.2
        if (feature_slice['vel_x'] >= 0.2).all():
            features_list.append(feature_slice.astype(np.float32).values)

            label_slice = time_slice.iloc[5][label_columns]
            label_list.append(label_slice.astype(np.float32).values)

            current_slice = time_slice.iloc[4][current_columns]
            current_list.append(current_slice.astype(np.float32).values)
    
    return features_list, label_list, current_list

if __name__ == "__main__":
    data_root = './data/FDM_dataset'
    csv_files = [f for f in os.listdir(data_root) if f.endswith('.csv')]

    all_features = []
    all_labels = []
    all_poses = []

    for csv_file in csv_files:
        print(f"processing: {csv_file}")
        filepath = os.path.join(data_root, csv_file)
        features, labels, poses = to_time_slice(filepath)
        all_features.extend(features)
        all_labels.extend(labels)
        all_poses.extend(poses)
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_poses = np.array(all_poses)

    # 额外的安全检查：确保没有vel_x < 0.2的样本通过
    # vel_x在feature中是第0列，检查所有时间步
    valid_mask = np.all(all_features[:, :, 0] >= 0.2, axis=1)
    all_features = all_features[valid_mask]
    all_labels = all_labels[valid_mask]
    all_poses = all_poses[valid_mask]
    
    print(f"After filtering vel_x < 0.2: {len(all_features)} samples remaining")

    features_train, features_temp, labels_train, labels_temp, poses_train, poses_temp = train_test_split(
        all_features, all_labels, all_poses, test_size=0.3, random_state=42)

    
    features_val, features_test, labels_val, labels_test, poses_val, poses_test = train_test_split(
        features_temp, labels_temp, poses_temp, test_size=0.5, random_state=42)

    np.savez_compressed('./data/training_data.npz', features=features_train, labels=labels_train, poses=poses_train)
    np.savez_compressed('./data/validation_data.npz', features=features_val, labels=labels_val, poses=poses_val)
    np.savez_compressed('./data/testing_data.npz', features=features_test, labels=labels_test, poses=poses_test)
    
    print("Data successfully split and saved!")
    print(f"Training set size: {len(features_train)}")
    print(f"Validation set size: {len(features_val)}")
    print(f"Test set size: {len(features_test)}")