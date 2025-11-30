import pandas as pd
import numpy as np
import os


def to_time_slice(filename):
    df = pd.read_csv(filename)

    window_size = 20
    step_size = 1
    min_brake = df['brake_sensor'].min()
    df['brake_sensor_scaled'] = (df['brake_sensor'] - min_brake) / (0.631 - min_brake)
    df['pedal_cc_scaled'] = df['pedal_cc'] / 1000
    
    label_columns = ['pedal_cc_scaled', 'steer_corrected', 'brake_sensor_scaled']
    condition_columns = ['pos_x', 'pos_y', 'heading']
    
    # 2Hz采样：假设原始数据是10Hz（每0.1秒一个样本），2Hz意味着每5个样本取一个
    # 对于condition数据，我们使用2Hz采样
    condition_sample_rate = 5  # 每5个样本取一个，实现2Hz采样
    
    label_list = []
    condition_list = []
    
    num_windows = (len(df) - window_size) // step_size + 1
    for i in range(num_windows):
        time_slice = df.iloc[i:i+window_size]
        
        # 获取20个时间步的label数据
        label_slice = time_slice[label_columns]
        label_list.append(label_slice.astype(np.float32).values)
        
        # condition使用2Hz采样，从20个时间步中每隔5个取一个样本
        condition_indices = range(0, window_size, condition_sample_rate)
        condition_slice = time_slice.iloc[condition_indices][condition_columns]
        condition_list.append(condition_slice.astype(np.float32).values)
    
    return label_list, condition_list


if __name__ == "__main__":
    data_root = '../data/FDM_dataset'
    csv_files = [f for f in os.listdir(data_root) if f.endswith('.csv')]

    all_labels = []
    all_conditions = []

    for csv_file in csv_files:
        print(f"processing: {csv_file}")
        filepath = os.path.join(data_root, csv_file)
        labels, conditions = to_time_slice(filepath)
        all_labels.extend(labels)
        all_conditions.extend(conditions)
    
    all_labels = np.array(all_labels)
    all_conditions = np.array(all_conditions)

    np.savez_compressed('../data/sampler_dataset.npz', 
                       labels=all_labels, conditions=all_conditions)
    
    print("Data successfully processed and saved!")
    print(f"Total dataset size: {len(all_labels)}")
    print(f"Label shape: {all_labels.shape}")
    print(f"Condition shape: {all_conditions.shape}")