"""
表示增强模块 (Representation Enhancement Modules)
实现基于信息几何和对比学习的多层次表示校正与相似度正则化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhiteningTransform(nn.Module):
    """
    白化变换模块 (Whitening Transform)
    消除BERT嵌入的主方向偏移,缓解表示塌陷
    
    公式: z' = (z - μ)(Σ + εI)^{-1/2}
    
    使用稳定的实现方式:
    1. 添加Tikhonov正则化到协方差矩阵
    2. 使用torch.linalg.svd替代torch.svd
    3. 过滤过小的奇异值
    4. 添加NaN/Inf检查
    """
    def __init__(self, eps=1e-3, momentum=0.9):  # 增大eps从1e-5到1e-3
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        # 用于存储运行统计量
        self.register_buffer('running_mean', None)
        self.register_buffer('running_cov', None)
        self.register_buffer('num_batches_tracked', torch.tensor(0))
        
    def forward(self, X, update_stats=True):
        """
        X: [batch, dim] - BERT输出嵌入
        返回白化后的表示
        """
        if X.dim() != 2:
            raise ValueError(f"Expected 2D tensor, got {X.dim()}D")
        
        batch_size, dim = X.shape
        
        if self.training and update_stats:
            # 计算批次统计量
            mu = X.mean(dim=0, keepdim=True)  # [1, dim]
            Xc = X - mu  # 中心化
            
            # 计算协方差矩阵,添加正则化项提高数值稳定性
            cov = (Xc.T @ Xc) / (batch_size - 1)  # [dim, dim]
            # 添加对角正则化 (Tikhonov regularization)
            cov = cov + self.eps * torch.eye(cov.size(0), device=cov.device)
            
            # 更新运行统计量
            if self.running_mean is None:
                self.running_mean = mu.squeeze(0).clone()
                self.running_cov = cov.clone()
            else:
                self.running_mean = (self.momentum * self.running_mean + 
                                   (1 - self.momentum) * mu.squeeze(0))
                self.running_cov = (self.momentum * self.running_cov + 
                                  (1 - self.momentum) * cov)
            
            self.num_batches_tracked += 1
            
            # 使用当前批次统计量
            use_mean = mu.squeeze(0)
            use_cov = cov
        else:
            # 推理时使用运行统计量
            if self.running_mean is None:
                # 如果没有统计量,跳过白化
                return X
            use_mean = self.running_mean
            use_cov = self.running_cov
        
        # 中心化
        Xc = X - use_mean.unsqueeze(0)
        
        # SVD分解获取白化矩阵 (使用更稳定的方法)
        try:
            # 尝试使用标准SVD
            U, S, _ = torch.linalg.svd(use_cov, full_matrices=False)
        except RuntimeError:
            # 如果失败,使用特征值分解作为fallback
            try:
                eigvals, eigvecs = torch.linalg.eigh(use_cov)
                # 取绝对值并排序(降序)
                eigvals = eigvals.abs()
                idx = torch.argsort(eigvals, descending=True)
                S = eigvals[idx]
                U = eigvecs[:, idx]
            except RuntimeError:
                # 如果还是失败,跳过白化,直接返回中心化结果
                print("警告: 白化变换失败,跳过此步骤")
                return Xc
        
        # 检查奇异值是否有效
        if torch.isnan(S).any() or torch.isinf(S).any():
            print("警告: 奇异值包含NaN/Inf,跳过白化")
            return Xc
        
        # 过滤过小的奇异值,提高数值稳定性
        # 使用更保守的阈值
        S_filtered = torch.clamp(S, min=self.eps * 10)
        
        # 白化矩阵 W = U @ diag(1/√(S+ε)) @ U^T
        # 计算逆平方根,并限制最大值防止数值爆炸
        D_inv_sqrt = 1.0 / torch.sqrt(S_filtered + self.eps)
        D_inv_sqrt = torch.clamp(D_inv_sqrt, max=100.0)  # 防止过大
        W = U @ torch.diag(D_inv_sqrt) @ U.T
        
        # 检查W是否包含NaN或Inf
        if torch.isnan(W).any() or torch.isinf(W).any():
            print("警告: 白化矩阵包含NaN/Inf,跳过白化")
            return Xc
        
        # 应用白化变换
        X_white = Xc @ W
        
        # 最后检查结果
        if torch.isnan(X_white).any() or torch.isinf(X_white).any():
            print("警告: 白化结果包含NaN/Inf,返回中心化结果")
            return Xc
        
        # 裁剪极端值,防止后续计算出现NaN
        X_white = torch.clamp(X_white, min=-10.0, max=10.0)
        
        return X_white


class IsotropyLoss(nn.Module):
    """
    各向同性正则损失 (Isotropy Regularization)
    增加余弦相似度分布方差,降低表示集中度
    
    公式: L_iso = 1/[N(N-1)] Σ_{i≠j} (cos(z_i,z_j) - μ0)^2
    """
    def __init__(self, target_mean=0.3):
        super().__init__()
        self.target_mean = target_mean
        
    def forward(self, Z):
        """
        Z: [batch, dim] - 归一化后的嵌入
        """
        # 检查输入
        if torch.isnan(Z).any() or torch.isinf(Z).any():
            return torch.tensor(0.0, device=Z.device)
        
        # 归一化
        Zn = F.normalize(Z, dim=-1, eps=1e-8)
        
        # 计算余弦相似度矩阵
        sim = Zn @ Zn.T  # [batch, batch]
        
        # 将对角线置零(排除自相似度)
        sim = sim.clone()
        sim.fill_diagonal_(0)
        
        # 计算损失
        loss = ((sim - self.target_mean) ** 2).mean()
        
        # 检查输出
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=Z.device)
        
        return loss


class SimilarityEntropyLoss(nn.Module):
    """
    相似度熵正则 (Similarity Entropy Regularization)
    最大化节点-样本相似度分布熵,防止相似度塌陷
    
    公式: 
    p_ij = exp(sim(z_i,s_j)/τ) / Σ_k exp(sim(z_i,s_k)/τ)
    L_ent = -1/N Σ_i Σ_j p_ij log p_ij
    """
    def __init__(self, temperature=0.7):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, Z_nodes, S_samples):
        """
        Z_nodes: [N, dim] - 节点嵌入
        S_samples: [M, dim] - 样本嵌入
        """
        # 检查输入
        if torch.isnan(Z_nodes).any() or torch.isinf(Z_nodes).any() or \
           torch.isnan(S_samples).any() or torch.isinf(S_samples).any():
            return torch.tensor(0.0, device=Z_nodes.device)
        
        # 归一化
        Z = F.normalize(Z_nodes, dim=-1, eps=1e-8)
        S = F.normalize(S_samples, dim=-1, eps=1e-8)
        
        # 计算相似度矩阵并应用温度
        sim = Z @ S.T / self.temperature  # [N, M]
        
        # 裁剪以防止softmax溢出
        sim = torch.clamp(sim, min=-50, max=50)
        
        # 计算softmax分布
        p = F.softmax(sim, dim=-1)  # [N, M]
        
        # 计算熵(取负号变为最小化目标)
        entropy = -(p * torch.log(p + 1e-9)).sum(dim=-1).mean()
        
        # 检查输出
        if torch.isnan(entropy) or torch.isinf(entropy):
            return torch.tensor(0.0, device=Z_nodes.device)
        
        # 返回负熵(最大化熵 = 最小化负熵)
        return -entropy


class StochasticNoise(nn.Module):
    """
    随机噪声增强 (Stochastic Noise Perturbation)
    增加局部扰动,避免过拟合高维子空间
    
    公式: z' = normalize(z/||z|| + ε·N(0,I))
    """
    def __init__(self, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std
        
    def forward(self, X, training=True):
        """
        X: [batch, dim] - 输入嵌入
        """
        if not training:
            return X
        
        # 归一化
        Xn = F.normalize(X, dim=-1)
        
        # 添加高斯噪声
        noise = torch.randn_like(Xn) * self.noise_std
        X_noisy = Xn + noise
        
        # 重新归一化
        return F.normalize(X_noisy, dim=-1)


class InfoNCELoss(nn.Module):
    """
    对比学习损失 (InfoNCE Contrastive Loss)
    强化正样本相似性,抑制负样本相似性
    
    公式: L_NCE = -1/N Σ_i log[exp(sim(z_i,s_i)/τ) / Σ_j exp(sim(z_i,s_j)/τ)]
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, Z, S, pos_idx=None):
        """
        Z: [N, dim] - 查询嵌入(如节点)
        S: [M, dim] - 键嵌入(如样本)
        pos_idx: [N] - 正样本索引,如果None则假设对角线为正样本
        """
        # 归一化
        Z = F.normalize(Z, dim=-1)
        S = F.normalize(S, dim=-1)
        
        # 计算相似度矩阵
        sim = Z @ S.T / self.temperature  # [N, M]
        
        if pos_idx is None:
            # 假设对角线为正样本(batch内对比)
            if Z.size(0) != S.size(0):
                raise ValueError("When pos_idx is None, Z and S must have same size")
            pos_idx = torch.arange(Z.size(0), device=Z.device)
        
        # 提取正样本相似度
        pos_sim = sim[torch.arange(sim.size(0), device=sim.device), pos_idx]
        
        # 计算InfoNCE损失
        loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(sim).sum(dim=-1) + 1e-9)
        ).mean()
        
        return loss


class TemperatureScaling(nn.Module):
    """
    温度标定模块 (Temperature Scaling)
    防止过高相似度进入softmax饱和区
    
    公式: sim'(a,b) = σ · cos(a,b)
    """
    def __init__(self, init_temp=0.5, learnable=True):
        super().__init__()
        if learnable:
            self.scale = nn.Parameter(torch.tensor(init_temp))
        else:
            self.register_buffer('scale', torch.tensor(init_temp))
        
    def forward(self, sim):
        """
        sim: [...] - 相似度矩阵或张量
        """
        return self.scale * sim


class LocalCalibration(nn.Module):
    """
    局部标度归一 (Local Calibration)
    减少局部强聚集导致的偏差
    
    公式: sim'_ij = sim_ij / (mean_{k∈N_K(i)} sim_ik + ε)
    """
    def __init__(self, k=5, eps=1e-9):
        super().__init__()
        self.k = k
        self.eps = eps
        
    def forward(self, sim_matrix):
        """
        sim_matrix: [N, M] - 相似度矩阵
        对每一行(节点)进行局部归一
        """
        k = min(self.k, sim_matrix.size(1))
        
        # 对每行取TopK
        vals, _ = torch.topk(sim_matrix, k=k, dim=-1)  # [N, k]
        
        # 计算局部均值
        local_mean = vals.mean(dim=-1, keepdim=True)  # [N, 1]
        
        # 局部归一化
        return sim_matrix / (local_mean + self.eps)


class DAGTopologyLoss(nn.Module):
    """
    DAG拓扑一致性损失 (DAG Topology Consistency)
    保持DAG中边权的语义一致性
    
    公式: L_dag = Σ_{(u,v)∈E} (sim(z_u,z_v) - w_uv)^2
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, Z, edge_index, edge_weights):
        """
        Z: [N, dim] - 节点嵌入
        edge_index: [2, E] - 边索引
        edge_weights: [E] - 边权重
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=Z.device)
        
        src, dst = edge_index[0], edge_index[1]
        
        # 计算边两端节点的余弦相似度
        Z_norm = F.normalize(Z, dim=-1)
        sim = F.cosine_similarity(Z_norm[src], Z_norm[dst], dim=-1)  # [E]
        
        # 计算与边权的MSE
        loss = ((sim - edge_weights) ** 2).mean()
        
        return loss


class EdgeSmoothnessLoss(nn.Module):
    """
    边权平滑正则 (Edge Smoothness Regularizer)
    强化高权重边的嵌入相似性
    
    公式: L_edge = Σ_{(u,v)∈E} ||z_u - z_v||^2 · (1 - w_uv)
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, Z, edge_index, edge_weights):
        """
        Z: [N, dim] - 节点嵌入
        edge_index: [2, E] - 边索引
        edge_weights: [E] - 边权重
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=Z.device)
        
        src, dst = edge_index[0], edge_index[1]
        
        # 计算边两端节点的欧式距离
        diff = (Z[src] - Z[dst]).pow(2).sum(dim=-1)  # [E]
        
        # 加权平滑损失(边权越高,惩罚越小)
        loss = (diff * (1 - edge_weights)).mean()
        
        return loss


class MultiLevelContrastiveLoss(nn.Module):
    """
    多层语义对比学习 (Multi-level Semantic Contrastive Learning)
    联合节点、样本、标签三者的多视角对比
    
    公式: L_multi = L(z_node, s_sample) + L(z_node, l_label) + L(s_sample, l_label)
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.nce = InfoNCELoss(temperature=temperature)
        
    def forward(self, Z_nodes, S_samples, L_labels, 
                node_label_idx=None, sample_label_idx=None):
        """
        Z_nodes: [N, dim] - 节点嵌入
        S_samples: [M, dim] - 样本嵌入
        L_labels: [K, dim] - 标签嵌入
        node_label_idx: [N] - 每个节点对应的标签索引
        sample_label_idx: [M] - 每个样本对应的标签索引
        """
        loss = 0.0
        
        # 1. 节点-标签对比
        if node_label_idx is not None:
            loss += self.nce(Z_nodes, L_labels, node_label_idx)
        
        # 2. 样本-标签对比
        if sample_label_idx is not None:
            loss += self.nce(S_samples, L_labels, sample_label_idx)
        
        # 3. 节点-样本对比(如果有对应关系)
        # 这里简化处理,可根据实际情况调整
        
        return loss


class VariationalInformationBottleneck(nn.Module):
    """
    变分信息瓶颈 (Variational Information Bottleneck)
    压缩无关噪声,提高任务无关泛化
    
    公式: L_VIB = D_KL(q(z|x) || N(0,I))
    """
    def __init__(self, dim, latent_dim=None):
        super().__init__()
        if latent_dim is None:
            latent_dim = dim
        
        self.encoder_mu = nn.Linear(dim, latent_dim)
        self.encoder_logvar = nn.Linear(dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, dim)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        """
        x: [batch, dim] - 输入表示
        返回: (z, kl_loss)
        """
        # 编码为均值和方差
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 计算KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        return z, kl_loss


class RepresentationEnhancementModule(nn.Module):
    """
    表示增强模块集成 (Integrated Representation Enhancement)
    整合所有表示校正和相似度正则化组件
    """
    def __init__(self, embed_dim, config=None):
        super().__init__()
        
        # 默认配置
        default_config = {
            'use_whitening': True,
            'use_isotropy': True,
            'use_similarity_entropy': True,
            'use_noise': True,
            'use_temperature': True,
            'use_local_calib': True,
            
            'whitening_eps': 1e-3,  # 增大以提高数值稳定性
            'isotropy_target': 0.3,
            'entropy_temperature': 0.7,
            'noise_std': 0.02,
            'temp_init': 0.5,
            'temp_learnable': True,
            'local_calib_k': 5,
        }
        
        if config is not None:
            default_config.update(config)
        
        self.config = default_config
        
        # 初始化模块
        if self.config['use_whitening']:
            self.whitening = WhiteningTransform(eps=self.config['whitening_eps'])
        
        if self.config['use_isotropy']:
            self.isotropy_loss = IsotropyLoss(target_mean=self.config['isotropy_target'])
        
        if self.config['use_similarity_entropy']:
            self.sim_entropy_loss = SimilarityEntropyLoss(
                temperature=self.config['entropy_temperature']
            )
        
        if self.config['use_noise']:
            self.noise = StochasticNoise(noise_std=self.config['noise_std'])
        
        if self.config['use_temperature']:
            self.temperature = TemperatureScaling(
                init_temp=self.config['temp_init'],
                learnable=self.config['temp_learnable']
            )
        
        if self.config['use_local_calib']:
            self.local_calib = LocalCalibration(k=self.config['local_calib_k'])
    
    def apply_whitening(self, X, update_stats=True):
        """应用白化变换"""
        if self.config['use_whitening']:
            return self.whitening(X, update_stats=update_stats)
        return X
    
    def apply_noise(self, X, training=True):
        """应用随机噪声"""
        if self.config['use_noise']:
            return self.noise(X, training=training)
        return X
    
    def compute_isotropy_loss(self, Z):
        """计算各向同性损失"""
        if self.config['use_isotropy']:
            return self.isotropy_loss(Z)
        return torch.tensor(0.0, device=Z.device)
    
    def compute_similarity_entropy_loss(self, Z_nodes, S_samples):
        """计算相似度熵损失"""
        if self.config['use_similarity_entropy']:
            return self.sim_entropy_loss(Z_nodes, S_samples)
        return torch.tensor(0.0, device=Z_nodes.device)
    
    def apply_temperature_scaling(self, sim):
        """应用温度标定"""
        if self.config['use_temperature']:
            return self.temperature(sim)
        return sim
    
    def apply_local_calibration(self, sim_matrix):
        """应用局部标度归一"""
        if self.config['use_local_calib']:
            return self.local_calib(sim_matrix)
        return sim_matrix

