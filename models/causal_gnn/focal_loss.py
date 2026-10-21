"""
Focal Loss for addressing class imbalance in multi-class classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡问题
    
    公式: FL(pt) = -α * (1-pt)^γ * log(pt)
    
    其中:
    - pt: 预测正确类别的概率
    - α: 平衡因子，用于平衡正负样本
    - γ: 聚焦参数，用于降低易分类样本的权重
    
    当γ=0时，退化为交叉熵损失
    当γ>0时，难分类样本（pt小）的权重增加
    
    适用场景:
    - 多类别分类，类别数>=4
    - 存在类别不平衡
    - 希望模型更关注难分类样本
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: 平衡因子，默认0.25
            gamma: 聚焦参数，默认2.0 (原论文推荐值)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_classes) 模型输出logits
            labels: (batch_size,) 真实标签索引
        
        Returns:
            loss: focal loss
        """
        # 1. 计算交叉熵损失 (不进行reduction)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)
        
        # 2. 计算pt (预测正确类别的概率)
        pt = torch.exp(-ce_loss)  # (B,)
        
        # 3. 计算focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # (B,)
        
        # 4. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss: 每个类别使用不同的α值
    
    适用于严重类别不平衡的场景
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: list or tensor of shape (num_classes,)
                   每个类别的权重，如果为None则自动计算
            gamma: 聚焦参数
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_classes)
            labels: (batch_size,)
        
        Returns:
            loss: balanced focal loss
        """
        num_classes = logits.size(1)
        device = logits.device
        
        # 1. 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, labels, reduction='none')  # (B,)
        
        # 2. 计算pt
        pt = torch.exp(-ce_loss)  # (B,)
        
        # 3. 如果没有提供alpha，自动计算类别权重
        if self.alpha is None:
            # 统计每个类别的样本数
            class_counts = torch.bincount(labels, minlength=num_classes).float()
            # 使用inverse frequency作为权重
            alpha = 1.0 / (class_counts + 1e-6)
            alpha = alpha / alpha.sum() * num_classes  # 归一化
        else:
            if isinstance(self.alpha, list):
                alpha = torch.tensor(self.alpha, device=device)
            else:
                alpha = self.alpha.to(device)
        
        # 4. 为每个样本选择对应的alpha
        alpha_t = alpha[labels]  # (B,)
        
        # 5. 计算balanced focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss  # (B,)
        
        # 6. Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

