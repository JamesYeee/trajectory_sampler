import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, shape, activation_fn, input_size, output_size, dropout=0.0, batchnorm=False):
        super(MLP, self).__init__()
        self.activation_fn = activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            if batchnorm:
                modules.append(nn.BatchNorm1d(shape[idx+1]))
            modules.append(self.activation_fn())
            if dropout != 0.0:
                modules.append(nn.Dropout(dropout))
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class Simplified_Trajectory_Sampler(nn.Module):
    """
    简化版轨迹采样器
    
    输入:
    - waypoints: 4个waypoints (2Hz × 2秒)
    
    输出:
    - action_sequence: 20个actions (10Hz × 2秒)
    """
    def __init__(self,
                 waypoint_dim,
                 action_dim,
                 latent_dim=64,
                 hidden_dims=[256, 256],
                 device='cuda'):
        super(Simplified_Trajectory_Sampler, self).__init__()
        
        self.waypoint_dim = waypoint_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # 固定序列长度
        self.waypoint_seq_len = 4   # 2Hz × 2秒
        self.action_seq_len = 20    # 10Hz × 2秒
        
        # waypoint编码器
        self.waypoint_encoder = MLP(
            shape=hidden_dims,
            activation_fn=nn.ReLU,
            input_size=waypoint_dim,
            output_size=latent_dim//2
        )
        
        # waypoint序列编码器 (简单的GRU)
        self.waypoint_gru = nn.GRU(
            input_size=latent_dim//2,
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 潜在空间编码器 (用于训练时的VAE)
        self.latent_mean_encoder = MLP(
            shape=[128, 128],
            activation_fn=nn.ReLU,
            input_size=latent_dim,
            output_size=latent_dim
        )
        
        self.latent_logvar_encoder = MLP(
            shape=[128, 128],
            activation_fn=nn.ReLU,
            input_size=latent_dim,
            output_size=latent_dim
        )
        
        # 解码器
        self.action_decoder_gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dims[0],
            num_layers=2,
            batch_first=True
        )
        
        self.action_output = MLP(
            shape=[hidden_dims[0]//2],
            activation_fn=nn.ReLU,
            input_size=hidden_dims[0],
            output_size=action_dim
        )

    def encode(self, waypoints):
        """
        编码waypoints
        
        Args:
            waypoints: (batch_size, 4, waypoint_dim)
        
        Returns:
            encoded_features: (batch_size, latent_dim)
        """
        # 编码waypoints
        batch_size = waypoints.shape[0]
        waypoints_flat = waypoints.view(-1, self.waypoint_dim)  # (batch_size*4, waypoint_dim)
        encoded_waypoints = self.waypoint_encoder.architecture(waypoints_flat)  # (batch_size*4, latent_dim//2)
        encoded_waypoints = encoded_waypoints.view(batch_size, 4, -1)  # (batch_size, 4, latent_dim//2)
        
        # 通过GRU处理waypoint序列
        _, waypoint_hidden = self.waypoint_gru(encoded_waypoints)  # (1, batch_size, latent_dim)
        encoded_waypoints = waypoint_hidden.squeeze(0)  # (batch_size, latent_dim)
        
        return encoded_waypoints

    def reparameterize(self, mu, logvar):
        """VAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_code):
        """
        从潜在编码解码出action序列
        
        Args:
            latent_code: (batch_size, latent_dim)
        
        Returns:
            actions: (batch_size, 20, action_dim)
        """
        batch_size = latent_code.shape[0]
        
        # 将潜在编码作为初始隐状态
        hidden = latent_code.unsqueeze(0).repeat(2, 1, 1)  # (2, batch_size, latent_dim)
        
        # 创建输入序列 (可以是零输入或学习的嵌入)
        decoder_input = latent_code.unsqueeze(1).repeat(1, self.action_seq_len, 1)  # (batch_size, 20, latent_dim)
        
        # 通过GRU解码
        gru_output, _ = self.action_decoder_gru(decoder_input, hidden)  # (batch_size, 20, hidden_dim)
        
        # 输出actions
        actions = self.action_output.architecture(gru_output.contiguous().view(-1, gru_output.shape[-1]))  # (batch_size*20, action_dim)
        actions = actions.view(batch_size, self.action_seq_len, self.action_dim)  # (batch_size, 20, action_dim)
        
        return actions

    def forward(self, waypoints, ground_truth_actions=None, training=True):
        """
        前向传播
        
        Args:
            waypoints: (batch_size, 4, waypoint_dim)
            ground_truth_actions: (batch_size, 20, action_dim) - 仅训练时需要
            training: bool
        
        Returns:
            如果training=True: (mu, logvar, reconstructed_actions, sampled_actions)
            如果training=False: sampled_actions
        """
        # 编码
        encoded_features = self.encode(waypoints)
        
        if training:
            # 训练模式：使用VAE损失
            mu = self.latent_mean_encoder.architecture(encoded_features)
            logvar = self.latent_logvar_encoder.architecture(encoded_features)
            
            # 重参数化采样
            latent_code = self.reparameterize(mu, logvar)
            
            # 解码
            reconstructed_actions = self.decode(latent_code)
            
            # 额外采样用于多样性
            sampled_latent = self.reparameterize(mu, logvar)
            sampled_actions = self.decode(sampled_latent)
            
            return mu, logvar, reconstructed_actions, sampled_actions
        else:
            encoded_features = self.encode(waypoints)
            # 可以从先验分布采样或使用编码特征
            latent_code = torch.randn(encoded_features.shape[0], self.latent_dim).to(self.device)
            # 或者使用编码特征: latent_code = encoded_features
            
            sampled_actions = self.decode(latent_code)
            return sampled_actions

    def sample_multiple(self, waypoints, n_samples=10):
        """
        生成多个可能的action序列
        
        Args:
            waypoints: (batch_size, 4, waypoint_dim)
            n_samples: int
        
        Returns:
            actions: (batch_size, n_samples, 20, action_dim)
        """
        self.eval()
        with torch.no_grad():
            batch_size = waypoints.shape[0]
            all_actions = []
            
            for _ in range(n_samples):
                # 每次采样不同的潜在编码
                encoded_features = self.encode(waypoints)
                latent_code = torch.randn(batch_size, self.latent_dim).to(self.device)
                actions = self.decode(latent_code)
                all_actions.append(actions.unsqueeze(1))
            
            return torch.cat(all_actions, dim=1)  # (batch_size, n_samples, 20, action_dim)



def compute_vae_loss(mu, logvar, reconstructed_actions, ground_truth_actions, kl_weight=0.001):
    """
    计算VAE损失
    
    Args:
        mu, logvar: 潜在分布参数
        reconstructed_actions: 重构的actions
        ground_truth_actions: 真实actions
        kl_weight: KL散度权重
    
    Returns:
        total_loss, reconstruction_loss, kl_loss
    """
    # 重构损失 (MSE)
    reconstruction_loss = F.mse_loss(reconstructed_actions, ground_truth_actions, reduction='mean')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    
    # 总损失
    total_loss = reconstruction_loss + kl_weight * kl_loss
    
    return total_loss, reconstruction_loss, kl_loss