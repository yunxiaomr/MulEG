"""
任务自适应Dropout模块
根据任务类型动态调整dropout率
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TaskAdaptiveDropout(nn.Module):
    """
    根据任务类型自适应调整dropout率
    
    不同任务类型的dropout策略：
    - semantic (语义): 高dropout (0.20-0.25) 防止过拟合
    - numerical (数值): 低dropout (0.05-0.10) 保留精确信息
    - ordinal (序数): 低dropout (0.10-0.15)
    - symbolic (符号): 低dropout (0.10-0.15)
    - mixed (混合): 中等dropout (0.15-0.20)
    """
    
    def __init__(self, default_dropout=0.15, use_spatial_dropout=True):
        super().__init__()
        self.default_dropout = default_dropout
        self.use_spatial_dropout = use_spatial_dropout
        self.task_dropout_map = {}
        
        # 预定义的dropout策略
        self.dropout_rules = {
            'semantic': 0.25,    # 语义任务：高dropout防止过拟合
            'numerical': 0.05,   # 数值任务：低dropout保留精确信息
            'ordinal': 0.10,     # 序数任务：低dropout
            'symbolic': 0.10,    # 符号任务：低dropout
            'mixed': 0.15        # 混合任务：中等dropout
        }
        
        # 当前激活的dropout率
        self.current_dropout_rate = default_dropout
        self.current_task_name = None
        self.current_task_type = None
    
    def set_task_dropout(self, task_name: str, task_type: str):
        """
        根据任务类型设置dropout
        
        Args:
            task_name: 任务名称
            task_type: 'semantic', 'numerical', 'ordinal', 'symbolic', 'mixed'
        """
        dropout = self.dropout_rules.get(task_type, self.default_dropout)
        self.task_dropout_map[task_name] = dropout
        self.current_dropout_rate = dropout
        self.current_task_name = task_name
        self.current_task_type = task_type
    
    def get_dropout(self, task_name: Optional[str] = None) -> float:
        """获取指定任务的dropout率"""
        if task_name is None:
            task_name = self.current_task_name
        
        if task_name is None:
            return self.default_dropout
        
        return self.task_dropout_map.get(task_name, self.default_dropout)
    
    def forward(self, x: torch.Tensor, task_name: Optional[str] = None) -> torch.Tensor:
        """
        应用任务自适应dropout
        
        Args:
            x: 输入张量 (batch_size, ..., feature_dim)
            task_name: 任务名称（可选）
        
        Returns:
            dropped: dropout后的张量
        """
        if not self.training:
            return x
        
        dropout_rate = self.get_dropout(task_name)
        
        if dropout_rate == 0.0:
            return x
        
        # 对数值特征使用SpatialDropout (丢弃整个特征通道)
        if self.use_spatial_dropout and (self.current_task_type in ['numerical', 'ordinal']):
            # SpatialDropout: 沿着特征维度进行dropout
            # 对于3D输入 (B, seq_len, D)，我们在D维度上做dropout
            if x.dim() == 3:
                # (B, L, D) -> (B, D, L) -> dropout -> (B, D, L) -> (B, L, D)
                x_perm = x.permute(0, 2, 1)
                dropped = F.dropout2d(x_perm.unsqueeze(-1), p=dropout_rate, training=True)
                dropped = dropped.squeeze(-1).permute(0, 2, 1)
                return dropped
            elif x.dim() == 2:
                # (B, D) -> (B, D, 1, 1) -> dropout2d -> (B, D, 1, 1) -> (B, D)
                x_expand = x.unsqueeze(-1).unsqueeze(-1)
                dropped = F.dropout2d(x_expand, p=dropout_rate, training=True)
                return dropped.squeeze(-1).squeeze(-1)
        
        # 标准dropout
        return F.dropout(x, p=dropout_rate, training=True)


class DynamicDropoutLayer(nn.Module):
    """
    动态Dropout层 - 可以在forward时指定dropout率
    """
    
    def __init__(self, dropout_controller: TaskAdaptiveDropout):
        super().__init__()
        self.dropout_controller = dropout_controller
    
    def forward(self, x: torch.Tensor, task_name: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量
            task_name: 任务名称
        
        Returns:
            dropped: dropout后的张量
        """
        return self.dropout_controller(x, task_name)


def load_task_types(task_type_json_path: str) -> Dict[str, str]:
    """
    从JSON文件加载任务类型映射
    
    Args:
        task_type_json_path: task_types_detailed.json文件路径
    
    Returns:
        task_type_map: Dict[task_name, task_type]
    """
    import json
    
    try:
        with open(task_type_json_path, 'r') as f:
            task_types = json.load(f)
    except:
        return {}
    
    # 反转映射: {task_name: task_type}
    task_type_map = {}
    for task_type, task_list in task_types.items():
        for task_name in task_list:
            task_type_map[task_name] = task_type
    
    return task_type_map


def initialize_task_dropouts(dropout_controller: TaskAdaptiveDropout, 
                             task_type_json_path: str):
    """
    初始化所有任务的dropout设置
    
    Args:
        dropout_controller: TaskAdaptiveDropout实例
        task_type_json_path: task_types_detailed.json文件路径
    """
    task_type_map = load_task_types(task_type_json_path)
    
    for task_name, task_type in task_type_map.items():
        dropout_controller.set_task_dropout(task_name, task_type)
    
    print(f"✅ 已为 {len(task_type_map)} 个任务初始化自适应dropout")
    
    # 打印统计信息
    type_counts = {}
    for task_type in task_type_map.values():
        type_counts[task_type] = type_counts.get(task_type, 0) + 1
    
    print(f"📊 任务类型分布:")
    for task_type, count in sorted(type_counts.items()):
        dropout_rate = dropout_controller.dropout_rules.get(task_type, 
                                                            dropout_controller.default_dropout)
        print(f"   {task_type}: {count}个任务, dropout={dropout_rate:.2f}")


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("任务自适应Dropout测试")
    print("=" * 60)
    
    # 创建dropout控制器
    dropout_controller = TaskAdaptiveDropout(default_dropout=0.15)
    
    # 测试不同任务类型
    test_cases = [
        ('occupancy-detection', 'numerical'),
        ('car-evaluation', 'ordinal'),
        ('tic-tac-toe-endgame', 'symbolic'),
        ('contraceptive-method-choice', 'mixed'),
        ('mushroom', 'semantic'),
    ]
    
    print("\n设置任务dropout:")
    for task_name, task_type in test_cases:
        dropout_controller.set_task_dropout(task_name, task_type)
        dropout_rate = dropout_controller.get_dropout(task_name)
        print(f"  {task_name} ({task_type}): dropout={dropout_rate:.2f}")
    
    # 测试dropout应用
    print("\n" + "=" * 60)
    print("Dropout应用测试")
    print("=" * 60)
    
    dropout_controller.train()
    
    for task_name, task_type in test_cases[:3]:
        dropout_controller.set_task_dropout(task_name, task_type)
        
        # 测试2D输入
        x_2d = torch.randn(4, 768)
        dropped_2d = dropout_controller(x_2d, task_name)
        
        # 测试3D输入
        x_3d = torch.randn(4, 10, 768)
        dropped_3d = dropout_controller(x_3d, task_name)
        
        # 计算dropout比例
        dropout_ratio_2d = (dropped_2d == 0).float().mean().item()
        dropout_ratio_3d = (dropped_3d == 0).float().mean().item()
        
        print(f"\n{task_name} ({task_type}):")
        print(f"  2D输入 shape: {x_2d.shape} -> {dropped_2d.shape}")
        print(f"  实际dropout率: {dropout_ratio_2d:.3f}")
        print(f"  3D输入 shape: {x_3d.shape} -> {dropped_3d.shape}")
        print(f"  实际dropout率: {dropout_ratio_3d:.3f}")

