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
    CVAE 轨迹采样器
    
    训练时: waypoints + ground_truth_actions → Encoder → Latent → Decoder → reconstructed_actions
    推理时: waypoints + sampled_latent → Decoder → sampled_actions
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
        
        # === 修改 1: 直接使用GRU编码waypoints (移除MLP) ===
        self.waypoint_gru = nn.GRU(
            input_size=waypoint_dim,  # 直接输入原始waypoint维度
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        )
        
        # === 修改 2: 直接使用GRU编码actions (移除MLP) ===
        self.action_gru = nn.GRU(
            input_size=action_dim,  # 直接输入原始action维度
            hidden_size=latent_dim,
            num_layers=1,
            batch_first=True
        )
        
        # === 保持不变: 潜在空间编码器 ===
        # 拼接后使用MLP解码出均值和标准差
        self.latent_mean_encoder = MLP(
            shape=[128, 128],
            activation_fn=nn.ReLU,
            input_size=latent_dim * 2,  # 拼接后的维度
            output_size=latent_dim
        )
        
        self.latent_logvar_encoder = MLP(
            shape=[128, 128],
            activation_fn=nn.ReLU,
            input_size=latent_dim * 2,  # 拼接后的维度
            output_size=latent_dim
        )
        
        # === 保持不变: 解码器 ===
        # 第一步：MLP处理拼接向量
        self.decoder_mlp = MLP(
            shape=[hidden_dims[0]],
            activation_fn=nn.ReLU,
            input_size=latent_dim * 2,  # 拼接后的维度
            output_size=hidden_dims[0]   # 输出到hidden_dim
        )

        # 第二步：GRU生成action序列
        self.action_decoder_gru = nn.GRU(
            input_size=hidden_dims[0],   # 从MLP输出的维度
            hidden_size=hidden_dims[0],  # 保持相同维度
            num_layers=3,                # 如图所示，3层GRU
            batch_first=True
        )

        # 第三步：输出层（简单的线性映射）
        self.action_output = nn.Linear(hidden_dims[0], action_dim)

    def encode_waypoints(self, waypoints):
        """直接使用GRU编码waypoints"""
        # waypoints: (batch_size, waypoint_seq_len, waypoint_dim)
        _, waypoint_hidden = self.waypoint_gru(waypoints)
        return waypoint_hidden.squeeze(0)  # (batch_size, latent_dim)

    def encode_actions(self, actions):
        """直接使用GRU编码actions"""
        # actions: (batch_size, action_seq_len, action_dim)
        _, action_hidden = self.action_gru(actions)
        return action_hidden.squeeze(0)  # (batch_size, latent_dim)

    def encode(self, waypoints, actions=None):
        """
        编码waypoints (和可选的actions)
        
        Args:
            waypoints: (batch_size, 4, waypoint_dim)
            actions: (batch_size, 20, action_dim) - 仅训练时提供
        
        Returns:
            encoded_features: (batch_size, latent_dim*2) 如果有actions
                            (batch_size, latent_dim) 如果只有waypoints
        """
        waypoint_encoding = self.encode_waypoints(waypoints)
        
        if actions is not None:
            # 训练模式：拼接 waypoints 和 actions 的编码
            action_encoding = self.encode_actions(actions)
            combined_encoding = torch.cat([waypoint_encoding, action_encoding], dim=-1)
            return combined_encoding, waypoint_encoding
        else:
            # 推理模式：只使用 waypoints
            return waypoint_encoding, waypoint_encoding

    def reparameterize(self, mu, logvar):
        """CVAE重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent_code, waypoint_encoding):
        """
        从潜在编码解码出action序列
        
        Args:
            latent_code: (batch_size, latent_dim)
            waypoint_encoding: (batch_size, latent_dim) - 条件信息
        
        Returns:
            actions: (batch_size, 20, action_dim)
        """
        batch_size = latent_code.shape[0]
        
        # 步骤1: 拼接潜在编码和条件信息
        combined = torch.cat([latent_code, waypoint_encoding], dim=-1)  
        # (batch_size, latent_dim*2)
        
        # 步骤2: 通过MLP处理拼接向量
        mlp_output = self.decoder_mlp.architecture(combined)  
        # (batch_size, hidden_dims[0])
        
        # 步骤3: 复制为序列输入
        decoder_input = mlp_output.unsqueeze(1).repeat(1, self.action_seq_len, 1)
        # (batch_size, 20, hidden_dims[0])
        
        # 步骤4: 通过GRU解码
        # 使用zero初始隐状态（如图中的<zero>标记）
        gru_output, _ = self.action_decoder_gru(decoder_input)
        # (batch_size, 20, hidden_dims[0])
        
        # 步骤5: 直接输出actions
        actions = self.action_output(gru_output)
        # (batch_size, 20, action_dim)
        
        return actions

    def forward(self, waypoints, ground_truth_actions=None, training=True, temperature=1.0):
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
        if training:
            # 训练模式：使用 waypoints + actions 进行编码
            combined_encoding, waypoint_encoding = self.encode(waypoints, ground_truth_actions)
            
            mu = self.latent_mean_encoder.architecture(combined_encoding)
            logvar = self.latent_logvar_encoder.architecture(combined_encoding)
            
            # 重参数化采样
            latent_code = self.reparameterize(mu, logvar)
            
            # 解码 (使用 waypoint_encoding 作为条件)
            reconstructed_actions = self.decode(latent_code, waypoint_encoding)
            
            # 额外采样
            sampled_latent = self.reparameterize(mu, logvar)
            sampled_actions = self.decode(sampled_latent, waypoint_encoding)
            
            return mu, logvar, reconstructed_actions, sampled_actions
        else:
            # 推理模式：只使用 waypoints，从先验分布采样
            waypoint_encoding, _ = self.encode(waypoints, actions=None)
            
            # 从标准正态分布采样，使用温度参数
            latent_code = torch.randn(waypoint_encoding.shape[0], self.latent_dim).to(self.device) * temperature
            
            # 解码
            sampled_actions = self.decode(latent_code, waypoint_encoding)
            return sampled_actions

    def sample_multiple(self, waypoints, n_samples=10, temperature=5.0):
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
            waypoint_encoding, _ = self.encode(waypoints, actions=None)
            
            all_actions = []
            for _ in range(n_samples):
                # 每次从先验分布采样不同的潜在编码
                latent_code = torch.randn(batch_size, self.latent_dim).to(self.device) * temperature
                actions = self.decode(latent_code, waypoint_encoding)
                all_actions.append(actions.unsqueeze(1))
            
            return torch.cat(all_actions, dim=1)  # (batch_size, n_samples, 20, action_dim)


def compute_cvae_loss(mu, logvar, reconstructed_actions, ground_truth_actions, kl_weight=0.001):
    """
    计算CVAE损失
    
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