"""
数值感知对比学习损失
让数值相近的样本在embedding空间也接近
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NumericalContrastiveLoss(nn.Module):
    """
    考虑数值相似性的对比学习损失
    让数值相近的样本在embedding空间也接近
    """
    
    def __init__(self, temperature=0.07, numerical_weight=0.5):
        super().__init__()
        self.temperature = temperature
        self.numerical_weight = numerical_weight  # 数值相似性的权重
    
    def forward(self, embeddings: torch.Tensor, 
                numeric_values: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (batch_size, embed_dim) 样本嵌入
            numeric_values: (batch_size, num_features) 数值特征
            labels: (batch_size,) 标签
        
        Returns:
            loss: 对比损失
        """
        B = embeddings.size(0)
        device = embeddings.device
        
        if B < 2:
            return torch.tensor(0.0, device=device)
        
        # 1. 计算embedding相似度
        embeddings_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.t())  # (B, B)
        
        # 2. 计算数值特征相似度 (如果提供)
        if numeric_values is not None and numeric_values.size(1) > 0:
            # 归一化数值特征
            numeric_values_norm = F.normalize(numeric_values, dim=-1)
            numeric_sim = torch.matmul(numeric_values_norm, numeric_values_norm.t())  # (B, B)
            # 将相似度映射到[0, 1]
            numeric_sim = (numeric_sim + 1) / 2
        else:
            numeric_sim = torch.ones(B, B, device=device)
        
        # 3. 构建正负样本mask
        if labels is not None:
            labels_eq = labels.unsqueeze(1) == labels.unsqueeze(0)  # (B, B)
            pos_mask = labels_eq.float()  # 同标签为正样本
        else:
            # 如果没有标签，使用数值相似度高的作为正样本
            threshold = 0.7
            pos_mask = (numeric_sim > threshold).float()
        
        neg_mask = 1.0 - pos_mask  # 不同标签为负样本
        
        # 去除对角线（自己）
        mask_diag = torch.eye(B, device=device).bool()
        pos_mask = pos_mask.masked_fill(mask_diag, 0)
        neg_mask = neg_mask.masked_fill(mask_diag, 0)
        
        # 检查是否有正样本
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # 4. 加权：数值相似的样本给予更高权重
        weighted_pos_sim = sim_matrix * pos_mask * (1.0 + self.numerical_weight * numeric_sim)
        weighted_neg_sim = sim_matrix * neg_mask
        
        # 5. InfoNCE损失
        logits = weighted_pos_sim / self.temperature
        exp_neg_logits = torch.exp(weighted_neg_sim / self.temperature)
        
        # 对每个样本，计算其正样本的平均对比损失
        log_prob = logits - torch.log(exp_neg_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # 6. 平均
        num_pos = pos_mask.sum(dim=1)
        loss = -((log_prob * pos_mask).sum(dim=1) / (num_pos + 1e-8)).mean()
        
        return loss


class TaskContrastiveLoss(nn.Module):
    """
    跨任务对比学习
    保证不同任务中"语义相似样本"在embedding空间更近
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings_taskA: torch.Tensor, 
                embeddings_taskB: torch.Tensor, 
                labels_taskA: torch.Tensor, 
                labels_taskB: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings_taskA: (batch_A, embed_dim) 任务A的样本嵌入
            embeddings_taskB: (batch_B, embed_dim) 任务B的样本嵌入
            labels_taskA: (batch_A,) 任务A的标签
            labels_taskB: (batch_B,) 任务B的标签
        
        Returns:
            loss: 跨任务对比损失
        """
        B_A = embeddings_taskA.size(0)
        B_B = embeddings_taskB.size(0)
        device = embeddings_taskA.device
        
        if B_A < 1 or B_B < 1:
            return torch.tensor(0.0, device=device)
        
        # 归一化
        emb_A = F.normalize(embeddings_taskA, dim=-1)
        emb_B = F.normalize(embeddings_taskB, dim=-1)
        
        # 计算跨任务相似度
        cross_sim = torch.matmul(emb_A, emb_B.t())  # (B_A, B_B)
        
        # 构建正样本mask（同标签视为正样本）
        # 注意: 这里假设不同任务的标签空间可能不同
        # 可以用标签文本的语义相似度来定义正样本
        
        # 简化版: 只考虑标签索引完全相同的情况
        labels_A_expand = labels_taskA.unsqueeze(1)  # (B_A, 1)
        labels_B_expand = labels_taskB.unsqueeze(0)  # (1, B_B)
        
        pos_mask = (labels_A_expand == labels_B_expand).float()  # (B_A, B_B)
        
        # 检查是否有正样本
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        # InfoNCE损失
        logits = cross_sim / self.temperature
        exp_logits = torch.exp(logits)
        
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # 平均
        num_pos = pos_mask.sum(dim=1)
        loss = -((log_prob * pos_mask).sum(dim=1) / (num_pos + 1e-8)).mean()
        
        return loss


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("数值感知对比学习损失测试")
    print("=" * 60)
    
    # 创建损失函数
    loss_fn = NumericalContrastiveLoss(temperature=0.07, numerical_weight=0.5)
    
    # 模拟数据
    batch_size = 8
    embed_dim = 768
    num_features = 5
    num_classes = 3
    
    embeddings = torch.randn(batch_size, embed_dim)
    numeric_values = torch.randn(batch_size, num_features)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失
    loss = loss_fn(embeddings, numeric_values, labels)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Numeric features: {num_features}")
    print(f"Labels: {labels.tolist()}")
    print(f"Loss: {loss.item():.4f}")
    
    # 测试跨任务对比学习
    print("\n" + "=" * 60)
    print("跨任务对比学习损失测试")
    print("=" * 60)
    
    task_loss_fn = TaskContrastiveLoss(temperature=0.07)
    
    # 模拟两个任务的数据
    batch_A = 6
    batch_B = 8
    
    embeddings_A = torch.randn(batch_A, embed_dim)
    embeddings_B = torch.randn(batch_B, embed_dim)
    labels_A = torch.randint(0, num_classes, (batch_A,))
    labels_B = torch.randint(0, num_classes, (batch_B,))
    
    # 计算损失
    task_loss = task_loss_fn(embeddings_A, embeddings_B, labels_A, labels_B)
    
    print(f"\nTask A batch size: {batch_A}")
    print(f"Task B batch size: {batch_B}")
    print(f"Task A labels: {labels_A.tolist()}")
    print(f"Task B labels: {labels_B.tolist()}")
    print(f"Task contrastive loss: {task_loss.item():.4f}")

