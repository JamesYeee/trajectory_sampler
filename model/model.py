from torch import nn
from sklearn.preprocessing import StandardScaler
import torch
import yaml
import pickle
import numpy as np
from abc import abstractmethod
import os

# feature: v[t-4,t], u[t-4,t]
# label: v[t+1]
# current pose: x[t], v[t]



class FDMNet(nn.Module):
    def __init__(self, input_size=6, hidden_size=25, num_layers=3, output_size=18, sequence_length=5):
        """
        FDM Neural Network for predicting physical parameters
        
        Args:
            input_size: Number of input features (6: vel_x, vel_y, rotation_rate, pedal_cc, steer, brake)
            hidden_size: Hidden dimension of GRU layers (25)
            num_layers: Number of GRU layers (3)
            output_size: Number of physical parameters to predict (17)
            sequence_length: Length of input sequence (5)
        """
        super(FDMNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.sequence_length = sequence_length


        
        # GRU layers (3 layers, hidden_size=25)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # After GRU: (batch_size, sequence_length, hidden_size) -> flatten to (batch_size, sequence_length * hidden_size)
        flattened_size = sequence_length * hidden_size  # 5 * 25 = 125
        
        # Dense layers (3 layers)
        self.dense_layers = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.BatchNorm1d(128),  
            nn.Mish(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 64),  
            nn.BatchNorm1d(64),
            nn.Mish(),
            nn.Dropout(0.1),
            
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
        # Load parameter 
        self.param_ranges_dict = self._load_param_ranges()
        self.veh_params = self._load_veh_params()
        self._setup_param_scaling()
        
    def _load_param_ranges(self):
        """Load parameter ranges from YAML file"""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'model_params', 'FDM_params.yaml')
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        param_ranges = {}
        for param_name, param_info in config['PHY_PARAM'].items():
            param_ranges[param_name] = {
                'min': param_info['Min'],
                'max': param_info['Max']
            }
        return param_ranges

    def _load_veh_params(self):
        """Load vehicle parameters from YAML file"""
        yaml_path = os.path.join(os.path.dirname(__file__), '..', 'model_params', 'FDM_params.yaml')
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['VEH_PARAM']
    
    def _setup_param_scaling(self):
        
        param_names = ['Bf', 'Cf', 'Df', 'Ef', 'Br', 'Cr', 'Dr', 'Er', 
                    'Cm1', 'Cm2', 'Cr0', 'Cr2', 'Crb', 'Iz', 'Shf', 'Svf', 'Shr', 'Svr']
        
        min_vals = []
        ranges = []
        
        for param_name in param_names:
            min_val = self.param_ranges_dict[param_name]['min']
            max_val = self.param_ranges_dict[param_name]['max']
            min_vals.append(min_val)
            ranges.append(max_val - min_val)
        
        
        self.register_buffer('param_min_vals', torch.tensor(min_vals))
        self.register_buffer('param_ranges', torch.tensor(ranges))
        
    def forward(self, x, current_s_a):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Scaled physical parameters tensor of shape (batch_size, output_size)
        """
        # GRU forward pass
        # x: (batch_size, sequence_length, input_size) -> (batch_size, sequence_length, hidden_size)
        gru_out, _ = self.gru(x)
        
        # Flatten: (batch_size, sequence_length, hidden_size) -> (batch_size, sequence_length * hidden_size)
        flattened = gru_out.reshape(gru_out.size(0), -1)
        
        # Dense layers with sigmoid activation
        # Output: (batch_size, output_size) with values in [0, 1]
        normalized_params = self.dense_layers(flattened)
        
        # Scale from [0, 1] to actual parameter ranges
        scaled_params = self._scale_to_param_ranges(normalized_params)

        pred_state = self.physical_dynamics(scaled_params, current_s_a)
        
        return pred_state
    
    def _scale_to_param_ranges(self, normalized_params):
        
        return normalized_params * self.param_ranges + self.param_min_vals

    def physical_dynamics(self, params, current_s_a_tensor, Ts=0.1):
        """
        Args:
            params: (batch_size, 18) 物理参数
            current_s_a_tensor: (batch_size, 9) 当前状态 
                [pos_x, pos_y, heading, vel_x, vel_y, rotation_rate, pedal_cc, steer_corrected, brake]
        """
        Bf, Cf, Df, Ef, Br, Cr, Dr, Er, Cm1, Cm2, Cr0, Cr2, Crb, Iz, Shf, Svf, Shr, Svr = torch.unbind(params, dim=1)
        
        # 直接从张量中提取，避免字典查找
        pos_x = current_s_a_tensor[:, 0]
        pos_y = current_s_a_tensor[:, 1] 
        heading = current_s_a_tensor[:, 2]
        vel_x = current_s_a_tensor[:, 3]
        vel_y = current_s_a_tensor[:, 4]
        rotation_rate = current_s_a_tensor[:, 5]
        pedal_cc = current_s_a_tensor[:, 6]
        steer_corrected = current_s_a_tensor[:, 7]
        brake = current_s_a_tensor[:, 8]
        
        # 缓存常用计算
        lf = self.veh_params['lf']
        lr = self.veh_params['lr'] 
        mass = self.veh_params['mass']
        
        Frx = (Cm1 - Cm2 * vel_x) * pedal_cc - Cr0 - Cr2 * vel_x**2 - Crb * brake
        
        # 缓存分子计算
        alpha_f_numerator = vel_y + rotation_rate * lf
        alpha_r_numerator = rotation_rate * lr - vel_y
        
        alpha_f = steer_corrected - torch.atan2(alpha_f_numerator, vel_x) + Shf
        alpha_r = torch.atan2(alpha_r_numerator, vel_x) + Shr
        
        # 预计算三角函数
        sin_steer = torch.sin(steer_corrected)
        cos_steer = torch.cos(steer_corrected)
        
        # 轮胎力计算
        Ffy = Svf + Df * torch.sin(Cf * torch.atan(Bf * alpha_f - Ef * (Bf * alpha_f - torch.atan(Bf * alpha_f))))
        Fry = Svr + Dr * torch.sin(Cr * torch.atan(Br * alpha_r - Er * (Br * alpha_r - torch.atan(Br * alpha_r))))
        
        # 状态预测
        v_x_pred = vel_x + Ts * (
            (Frx - Ffy * sin_steer) / mass + vel_y * rotation_rate
        )
        
        v_y_pred = vel_y + Ts * (
            (Fry + Ffy * cos_steer) / mass - vel_x * rotation_rate
        )
        
        yaw_rate_pred = rotation_rate + Ts * (
            Ffy * lf * cos_steer - Fry * lr
        ) / Iz
        """
        ### only use v_x, v_y, yaw_rate as loss ###

        x_pred = current_s_a['pos_x'] + Ts * (
            current_s_a['vel_x'] * torch.cos(current_s_a['heading']) - 
            current_s_a['vel_y'] * torch.sin(current_s_a['heading'])
        )
        
        y_pred = current_s_a['pos_y'] + Ts * (
            current_s_a['vel_x'] * torch.sin(current_s_a['heading']) + 
            current_s_a['vel_y'] * torch.cos(current_s_a['heading'])
        )
        
        heading_pred = current_s_a['heading'] + Ts * current_s_a['rotation_rate']
        """
        
        pred_state = torch.stack([v_x_pred, v_y_pred, yaw_rate_pred], dim=1)
        return pred_state

    






