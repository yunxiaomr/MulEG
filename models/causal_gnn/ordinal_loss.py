"""
序回归损失 (Ordinal Regression Loss)
适用于类别有明确顺序的任务
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import List, Tuple, Dict, Optional


class OrdinalRegressionLoss(nn.Module):
    """
    序回归损失：适用于类别有明确顺序的任务
    例如: low < medium < high, 评分1 < 2 < 3 < 4 < 5
    
    包含四个损失项:
    1. 交叉熵损失 (保持分类能力)
    2. 排序MSE损失 (鼓励预测的期望rank接近真实rank)
    3. 距离惩罚 (预测错误时，距离越远惩罚越大)
    4. 类别相似度惩罚 (让预测分布接近以真实标签为中心的相似度分布)
    """
    
    def __init__(self, num_classes, alpha=0.5, beta=0.2, task_name=None, label_order=None):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha  # 排序损失权重 (降低从1.0到0.5)
        self.beta = beta    # 距离惩罚权重 (降低从0.5到0.2)
        self.task_name = task_name
        self.label_order = label_order  # 正确的标签顺序
        
        # 构建类别相似度矩阵: S_ij = 1 / (1 + |i - j|)
        self.register_buffer('similarity_matrix', self._build_similarity_matrix())
    
    def _build_similarity_matrix(self):
        """
        构建类别相似度矩阵: S_ij = 1 / (1 + |i - j|)
        """
        indices = torch.arange(self.num_classes).float()
        dist = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        sim_matrix = 1.0 / (1.0 + dist)
        return sim_matrix
    
    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, num_classes) 模型输出
            labels: (batch_size,) 真实标签索引
        
        Returns:
            loss: 序回归损失
            metrics: dict - 包含各个损失项的详细信息
        """
        batch_size = logits.size(0)
        device = logits.device
        
        # 1. 计算预测概率
        probs = F.softmax(logits, dim=-1)  # (B, C)
        
        # 2. 计算期望的类别索引 (排序MSE损失)
        class_indices = torch.arange(self.num_classes, device=device).float()
        predicted_ranks = (probs * class_indices).sum(dim=1)  # (B,)
        true_ranks = labels.float()  # (B,)
        rank_loss = F.mse_loss(predicted_ranks, true_ranks)
        
        # 3. 交叉熵损失（保持分类能力）
        ce_loss = F.cross_entropy(logits, labels)
        
        # 4. 距离惩罚（预测错误时，距离越远惩罚越大）
        pred_class = logits.argmax(dim=-1)  # (B,)
        distance = torch.abs(pred_class - labels).float()  # (B,)
        distance_penalty = (distance ** 2).mean()
        
        # 5. 类别相似度惩罚
        # 让预测分布接近"以真实标签为中心的相似度分布"
        onehot = F.one_hot(labels, num_classes=self.num_classes).float()  # (B, C)
        sim_matrix = self.similarity_matrix.to(device)  # (C, C)
        target_dist = onehot @ sim_matrix  # (B, C)
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)
        similarity_penalty = F.kl_div(
            torch.log(probs + 1e-8), 
            target_dist, 
            reduction='batchmean'
        )
        
        # 6. 总损失
        loss = (ce_loss 
                + self.alpha * rank_loss 
                + self.beta * distance_penalty
                + self.beta * similarity_penalty)
        
        metrics = {
            'ce_loss': ce_loss.item(),
            'rank_loss': rank_loss.item(),
            'distance_penalty': distance_penalty.item(),
            'similarity_penalty': similarity_penalty.item(),
            'total_loss': loss.item()
        }
        
        return loss, metrics


def detect_ordinal_labels(labels: List[str], task_name: str = None) -> Tuple[bool, List[str]]:
    """
    检测标签是否有序关系
    
    Args:
        labels: List[str] 标签列表
        task_name: str 任务名称（可选，用于特殊任务匹配）
    
    Returns:
        is_ordinal: bool 是否有序
        label_order: List[str] 排序后的标签（如果有序）
    """
    # 🔥 首先检查手动定义的正确序数顺序
    correct_ordinal_orders = {
        # 'student-performance': ['Average', 'Good', 'V.good', 'Excellent'],
        # 'car-evaluation': ['unacc', 'acc', 'good', 'vgood'],
        'teaching-assistant-evaluation': ['Low', 'Medium', 'High'],
        'hotel-rating': ['<=3', '>3'],
        'somerville-happiness': ['unhappy', 'happy'],
        'wine': ['1.0', '2.0', '3.0'],
    }
    
    if task_name and task_name in correct_ordinal_orders:
        correct_order = correct_ordinal_orders[task_name]
        # 验证所有标签都在正确顺序中
        if set(labels) == set(correct_order):
            print(f"✅ 使用手动定义的序数顺序 for {task_name}: {correct_order}")
            return True, correct_order
    
    # 1. 检查数值序
    try:
        numeric_labels = [float(l) for l in labels]
        sorted_labels = [str(l) for l in sorted(numeric_labels)]
        return True, sorted_labels
    except:
        pass
    
    # 2. 检查常见序关系模式 (扩展版，从real_task_details.json中提取)
    ordinal_patterns = {
        'size': ['tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge'],
        'quality': ['terrible', 'poor', 'fair', 'good', 'excellent', 'outstanding'],
        'level': ['vlow', 'very-low', 'low', 'med', 'medium', 'high', 'vhigh', 'very-high'],
        'grade': ['f', 'd', 'c', 'b', 'a', 'a+'],
        'rating': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'acceptance': ['unacc', 'acc', 'good', 'vgood'],  # car-evaluation
        'performance': ['average', 'good', 'v.good', 'excellent'],  # student-performance
        'teaching_quality': ['low', 'medium', 'high'],  # teaching-assistant-evaluation
        'happiness': ['unhappy', 'happy'],  # somerville-happiness
        'rating_range': ['<=3', '>3', '<=50%', '>50%', '<50', '>=50', '<=1', '>1'],  # various rating tasks
    }
    
    labels_lower = [l.lower().replace('_', '-').replace('.', '') for l in labels]
    
    for pattern_name, pattern in ordinal_patterns.items():
        pattern_lower = [p.lower().replace('_', '-').replace('.', '') for p in pattern]
        
        # 检查labels是否是pattern的子集
        if all(l in pattern_lower for l in labels_lower):
            # 按pattern顺序排序
            ordered_lower = [l for l in pattern_lower if l in labels_lower]
            
            # 映射回原始大小写
            label_map = {l.lower().replace('_', '-').replace('.', ''): orig 
                        for orig, l in zip(labels, labels)}
            ordered = [label_map[l] for l in ordered_lower]
            
            return True, ordered
    
    # 3. 任务特定的序关系
    task_specific_orders = {
        'car-evaluation': ['unacc', 'acc', 'good', 'vgood'],
        'student-performance': ['Average', 'Good', 'V.good', 'Excellent'],
        'teaching-assistant-evaluation': ['Low', 'Medium', 'High'],
        'hotel-rating': ['<=3', '>3'],
        'somerville-happiness': ['unhappy', 'happy'],
        'wine': ['1.0', '2.0', '3.0'],
        'color-luminance': ['<=50%', '>50%'],
        'coin-face-value': ['<=1', '>1'],
        'driving-championship-points': ['<50', '>=50'],
    }
    
    if task_name and task_name in task_specific_orders:
        expected_order = task_specific_orders[task_name]
        # 检查是否匹配
        if set(labels) == set(expected_order):
            return True, expected_order
    
    return False, labels


def create_ordinal_label_map(task_name: str, labels: List[str]) -> Optional[Dict[str, int]]:
    """
    为有序任务创建标签到索引的映射
    
    Args:
        task_name: 任务名称
        labels: 标签列表
    
    Returns:
        label_to_idx: Dict[str, int] 或 None（如果不是有序任务）
    """
    is_ordinal, ordered_labels = detect_ordinal_labels(labels, task_name)
    
    if is_ordinal:
        label_to_idx = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
        return label_to_idx
    
    return None


def get_ordinal_task_info(real_task_details_path: str = None) -> Dict[str, Dict]:
    """
    从real_task_details.json中提取所有有序任务的信息
    
    Args:
        real_task_details_path: real_task_details.json文件路径
    
    Returns:
        ordinal_info: Dict[task_name, {'labels': [...], 'ordered_labels': [...]}]
    """
    if real_task_details_path is None:
        real_task_details_path = 'clues_data/clues_real_utils/real_task_details.json'
    
    try:
        with open(real_task_details_path, 'r') as f:
            task_details = json.load(f)
    except:
        return {}
    
    ordinal_info = {}
    
    for task_name, details in task_details.items():
        labels = details.get('task_lbls', [])
        is_ordinal, ordered_labels = detect_ordinal_labels(labels, task_name)
        
        if is_ordinal:
            ordinal_info[task_name] = {
                'labels': labels,
                'ordered_labels': ordered_labels,
                'num_classes': len(labels)
            }
    
    return ordinal_info


# 测试代码
if __name__ == '__main__':
    # 测试序关系检测
    test_cases = [
        ('car-evaluation', ['unacc', 'acc', 'good', 'vgood']),
        ('student-performance', ['Excellent', 'V.good', 'Good', 'Average']),
        ('teaching-assistant-evaluation', ['Low', 'Medium', 'High']),
        ('wine', ['1.0', '2.0', '3.0']),
        ('mushroom', ['poisonous', 'edible']),  # 非有序
    ]
    
    print("=" * 60)
    print("序关系检测测试")
    print("=" * 60)
    
    for task_name, labels in test_cases:
        is_ordinal, ordered = detect_ordinal_labels(labels, task_name)
        print(f"\n任务: {task_name}")
        print(f"  原始标签: {labels}")
        print(f"  是否有序: {is_ordinal}")
        if is_ordinal:
            print(f"  排序结果: {ordered}")
    
    # 测试损失计算
    print("\n" + "=" * 60)
    print("序回归损失测试")
    print("=" * 60)
    
    num_classes = 4
    batch_size = 8
    loss_fn = OrdinalRegressionLoss(num_classes=num_classes, alpha=1.0, beta=0.5)
    
    # 模拟预测
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    loss, metrics = loss_fn(logits, labels)
    
    print(f"\n预测logits shape: {logits.shape}")
    print(f"真实标签: {labels.tolist()}")
    print(f"预测类别: {logits.argmax(dim=-1).tolist()}")
    print(f"\n损失详情:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

