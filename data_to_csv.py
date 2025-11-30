#get data from json

## get state: orientation, pos(xyz),rotation_rate,vel
##
## get input   
##              1.pedal(zeo vehicle info)
##              2.brake(zeo sensors)
##              3.steer_corrected


import json
import pandas as pd
import os
import math


def calculate_inertial_heading(orientation):

    q0 = orientation[0]
    q3 = orientation[3]
    return 2 * math.atan2(q3, q0)


def combine_and_to_csv(scenes_num, root_path):
    for scenes_id in range(1, scenes_num+1):
        ...

def save_pos_data(scene_id, root_path):
    
    file_path=os.path.join(root_path, 'data', 'can_bus')
    
    #get path
    if isinstance(scene_id, int):
        scene_id = f"{scene_id:04d}"
    filename_pose = f"scene-{scene_id}_pose.json"
    full_path_pose = os.path.join(file_path, filename_pose)

    #check exists
    if os.path.exists(full_path_pose):
        # Load the JSON data from the file
        with open(full_path_pose, 'r') as f:
            data = json.load(f)

        # Create a DataFrame
        df = pd.DataFrame(data)

        # get useful data
        utime_df = df[['utime']]
        pos_list = [item[:2] for item in df['pos'].tolist()]
        pos_df = pd.DataFrame(pos_list, columns=['pos_x', 'pos_y'])
        heading_values = [calculate_inertial_heading(item) for item in df['orientation'].tolist()]
        heading_df = pd.DataFrame(heading_values, columns=['heading'])
        vel_list = [item[:2] for item in df['vel'].tolist()]
        vel_df = pd.DataFrame(vel_list, columns=['vel_x', 'vel_y'])
        rotation_rate = [item[-1] for item in df['rotation_rate'].tolist()]
        ro_rate_df =  pd.DataFrame(rotation_rate, columns=['rotation_rate'])
        
        combined_df = pd.concat([utime_df, pos_df, heading_df, vel_df, ro_rate_df], axis=1)


        # Convert utime to datetime objects for resampling
        combined_df['utime'] = pd.to_datetime(combined_df['utime'], unit='us')

        # Set utime as the index
        combined_df.set_index('utime', inplace=True)

        #    Resample the data to a 40 millisecond ('40ms') frequency.
        #    Then, use interpolation to calculate the values for the new timestamps.
        #    'time' interpolation is ideal for unevenly spaced data (even though ours is nearly even).
        resampled_df = combined_df.resample('100ms').mean().interpolate(method='time')

        #    (Optional but recommended) Move the 'utime' index back to a column.
        resampled_df.reset_index(inplace=True)

        return resampled_df

def save_control_data(scene_id, root_path):

    file_path=os.path.join(root_path, 'data', 'can_bus')

    #get path
    if isinstance(scene_id, int):
        scene_id = f"{scene_id:04d}"
    filename_zoe = f"scene-{scene_id}_zoe_veh_info.json"
    full_path_zoe = os.path.join(file_path, filename_zoe)

    #check exists
    if os.path.exists(full_path_zoe):
        # Load the JSON data from the file
        with open(full_path_zoe, 'r') as f:
            data = json.load(f)
    
        df = pd.DataFrame(data)
        
        utime_df = df[['utime']]
        pedal_df = df[['pedal_cc']]
        steer_df = df[['steer_corrected']]

        combined_df = pd.concat([utime_df, pedal_df, steer_df], axis=1)
        combined_df['utime'] = pd.to_datetime(combined_df['utime'], unit='us')
        combined_df.set_index('utime', inplace=True)
        resampled_df = combined_df.resample('100ms').mean().interpolate(method='time')
        resampled_df.reset_index(inplace=True)

        return resampled_df
    
def save_brake_data(scene_id, root_path):
    file_path=os.path.join(root_path, 'data', 'can_bus')

    #get path
    if isinstance(scene_id, int):
        scene_id = f"{scene_id:04d}"
    filename_zoe = f"scene-{scene_id}_zoesensors.json"
    full_path_zoe = os.path.join(file_path, filename_zoe)

    #check exists
    if os.path.exists(full_path_zoe):
        # Load the JSON data from the file
        with open(full_path_zoe, 'r') as f:
            data = json.load(f)
    
        df = pd.DataFrame(data)
        
        utime_df = df[['utime']]
        brake_df = df[['brake_sensor']]
        combined_df = pd.concat([utime_df, brake_df], axis=1)
        combined_df['utime'] = pd.to_datetime(combined_df['utime'], unit='us')
        combined_df.set_index('utime', inplace=True)
        resampled_df = combined_df.resample('100ms').mean().interpolate(method='time')
        resampled_df.reset_index(inplace=True)

        return resampled_df

def merge_data(scene_id, root_path):

    pos_df = save_pos_data(scene_id, root_path)
    control_df = save_control_data(scene_id, root_path)
    brake_df = save_brake_data(scene_id, root_path)

    if pos_df is None or control_df is None or brake_df is None:
        print(f"Skipping scene {scene_id:04d} due to missing data.")
        return

    merged_df = pd.merge(pos_df, control_df, on='utime', how='inner')
    final_merged_df = pd.merge(merged_df, brake_df, on='utime', how='inner')

    output_filename = f'scene-{scene_id:04d}_merged_data.csv'
    output_path = os.path.join(root_path, 'data', 'FDM_dataset', output_filename)

    final_merged_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    
    root_path = os.getcwd()
    for test_id in range(1, 1111):

        merge_data(test_id, root_path)