"""
任务嵌入模块 (Task Embedding)
支持任务感知表示和轻量级Adapter调整
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class TaskEmbeddingModule(nn.Module):
    """
    为每个任务学习一个可学习的嵌入向量
    类似Meta-Learning中的任务表示
    """
    
    def __init__(self, num_tasks=36, embed_dim=768):
        super().__init__()
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        
        # 任务嵌入
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        
        # 任务ID映射
        self.task_name_to_id = {}
        self.task_id_to_name = {}
        self.next_task_id = 0
    
    def register_task(self, task_name: str, task_id: Optional[int] = None):
        """
        注册一个任务
        
        Args:
            task_name: 任务名称
            task_id: 任务ID（可选，不提供则自动分配）
        """
        if task_name in self.task_name_to_id:
            return  # 已注册
        
        if task_id is None:
            task_id = self.next_task_id
            self.next_task_id += 1
        
        self.task_name_to_id[task_name] = task_id
        self.task_id_to_name[task_id] = task_name
    
    def get_task_id(self, task_name: str) -> int:
        """获取任务ID"""
        if task_name not in self.task_name_to_id:
            self.register_task(task_name)
        return self.task_name_to_id[task_name]
    
    def forward(self, fused_emb: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        添加任务嵌入
        
        Args:
            fused_emb: (batch_size, embed_dim) 融合后的嵌入
            task_name: str 任务名称
        
        Returns:
            task_aware_emb: (batch_size, embed_dim) 任务感知嵌入
        """
        task_id = self.get_task_id(task_name)
        device = fused_emb.device
        
        # 获取任务嵌入
        task_id_tensor = torch.tensor([task_id], device=device)
        task_emb = self.task_embedding(task_id_tensor)  # (1, D)
        
        # 添加到样本嵌入
        task_aware_emb = fused_emb + task_emb  # (B, D)
        
        return task_aware_emb


class TaskAdapter(nn.Module):
    """
    为每个任务插入轻量适配模块
    不改变主干参数，只调整任务特定参数
    使用Bottleneck结构 (下投影 + 上投影)
    """
    
    def __init__(self, embed_dim=768, adapter_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        
        # 下投影 + 上投影 (Bottleneck结构)
        self.down_proj = nn.Linear(embed_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, embed_dim)
        self.activation = nn.GELU()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, embed_dim)
        
        Returns:
            adapted_x: (batch_size, embed_dim)
        """
        residual = x
        
        # Layer norm
        x = self.layer_norm(x)
        
        # Adapter前向
        x = self.down_proj(x)  # (B, adapter_dim)
        x = self.activation(x)
        x = self.up_proj(x)    # (B, embed_dim)
        
        # 残差连接
        return residual + x


class MultiTaskAdapterModule(nn.Module):
    """
    管理多个任务的Adapter
    每个任务有自己独立的Adapter参数
    """
    
    def __init__(self, num_tasks=36, embed_dim=768, adapter_dim=64):
        super().__init__()
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        
        # 为每个任务创建adapter
        self.adapters = nn.ModuleDict()
        
        # 任务映射
        self.task_name_to_id = {}
        self.task_id_to_name = {}
        self.next_task_id = 0
    
    def register_task(self, task_name: str, task_id: Optional[int] = None):
        """
        注册一个任务并创建其adapter
        
        Args:
            task_name: 任务名称
            task_id: 任务ID（可选）
        """
        if task_name in self.task_name_to_id:
            return  # 已注册
        
        if task_id is None:
            task_id = self.next_task_id
            self.next_task_id += 1
        
        self.task_name_to_id[task_name] = task_id
        self.task_id_to_name[task_id] = task_name
        
        # 创建adapter并移动到正确的设备
        adapter_key = f"task_{task_id}"
        adapter = TaskAdapter(self.embed_dim, self.adapter_dim)
        
        # 获取当前模块所在的设备（通过检查adapters ModuleDict的设备）
        try:
            # 如果已经有adapter，使用它们的设备
            if len(self.adapters) > 0:
                first_adapter = next(iter(self.adapters.values()))
                device = next(first_adapter.parameters()).device
            else:
                # 否则尝试从父模块获取
                device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        adapter = adapter.to(device)
        self.adapters[adapter_key] = adapter
    
    def get_task_id(self, task_name: str) -> int:
        """获取任务ID"""
        if task_name not in self.task_name_to_id:
            self.register_task(task_name)
        return self.task_name_to_id[task_name]
    
    def forward(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        应用任务特定的adapter
        
        Args:
            x: (batch_size, embed_dim)
            task_name: 任务名称
        
        Returns:
            adapted_x: (batch_size, embed_dim)
        """
        task_id = self.get_task_id(task_name)
        adapter_key = f"task_{task_id}"
        
        if adapter_key not in self.adapters:
            # 如果adapter不存在，直接返回原始输入
            return x
        
        adapter = self.adapters[adapter_key]
        return adapter(x)


class TaskEmbeddingWithAdapter(nn.Module):
    """
    结合任务嵌入和任务Adapter的完整模块
    """
    
    def __init__(self, num_tasks=36, embed_dim=768, adapter_dim=64, 
                 use_adapter=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_adapter = use_adapter
        
        # 任务嵌入
        self.task_embedding = TaskEmbeddingModule(num_tasks, embed_dim)
        
        # 任务Adapter
        if use_adapter:
            self.task_adapter = MultiTaskAdapterModule(num_tasks, embed_dim, adapter_dim)
    
    def register_task(self, task_name: str, task_id: Optional[int] = None):
        """注册任务"""
        self.task_embedding.register_task(task_name, task_id)
        if self.use_adapter:
            self.task_adapter.register_task(task_name, task_id)
    
    def forward(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        应用任务嵌入和adapter
        
        Args:
            x: (batch_size, embed_dim) 输入嵌入
            task_name: 任务名称
        
        Returns:
            output: (batch_size, embed_dim) 任务感知输出
        """
        # 1. 添加任务嵌入
        x = self.task_embedding(x, task_name)
        
        # 2. 应用任务adapter
        if self.use_adapter:
            x = self.task_adapter(x, task_name)
        
        return x


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("任务嵌入模块测试")
    print("=" * 60)
    
    num_tasks = 36
    embed_dim = 768
    adapter_dim = 64
    batch_size = 8
    
    # 创建任务嵌入模块
    task_emb_module = TaskEmbeddingModule(num_tasks, embed_dim)
    
    # 注册一些任务
    tasks = [
        'occupancy-detection',
        'car-evaluation',
        'tic-tac-toe-endgame',
        'mushroom'
    ]
    
    for task in tasks:
        task_emb_module.register_task(task)
    
    print(f"\n已注册任务:")
    for task in tasks:
        task_id = task_emb_module.get_task_id(task)
        print(f"  {task}: ID={task_id}")
    
    # 测试前向传播
    print("\n" + "=" * 60)
    print("前向传播测试")
    print("=" * 60)
    
    for task in tasks[:2]:
        x = torch.randn(batch_size, embed_dim)
        output = task_emb_module(x, task)
        print(f"\n任务: {task}")
        print(f"  输入shape: {x.shape}")
        print(f"  输出shape: {output.shape}")
        print(f"  输入输出差异范数: {torch.norm(output - x).item():.4f}")
    
    # 测试Adapter
    print("\n" + "=" * 60)
    print("TaskAdapter测试")
    print("=" * 60)
    
    adapter = TaskAdapter(embed_dim, adapter_dim)
    x = torch.randn(batch_size, embed_dim)
    adapted = adapter(x)
    
    print(f"\n输入shape: {x.shape}")
    print(f"输出shape: {adapted.shape}")
    print(f"残差范数: {torch.norm(adapted - x).item():.4f}")
    
    # 测试MultiTaskAdapter
    print("\n" + "=" * 60)
    print("MultiTaskAdapterModule测试")
    print("=" * 60)
    
    multi_adapter = MultiTaskAdapterModule(num_tasks, embed_dim, adapter_dim)
    
    for task in tasks:
        multi_adapter.register_task(task)
    
    for task in tasks[:2]:
        x = torch.randn(batch_size, embed_dim)
        output = multi_adapter(x, task)
        print(f"\n任务: {task}")
        print(f"  输入shape: {x.shape}")
        print(f"  输出shape: {output.shape}")
        print(f"  残差范数: {torch.norm(output - x).item():.4f}")
    
    # 测试完整模块
    print("\n" + "=" * 60)
    print("TaskEmbeddingWithAdapter完整测试")
    print("=" * 60)
    
    full_module = TaskEmbeddingWithAdapter(num_tasks, embed_dim, adapter_dim, use_adapter=True)
    
    for task in tasks:
        full_module.register_task(task)
    
    for task in tasks[:2]:
        x = torch.randn(batch_size, embed_dim)
        output = full_module(x, task)
        print(f"\n任务: {task}")
        print(f"  输入shape: {x.shape}")
        print(f"  输出shape: {output.shape}")
        print(f"  输入输出差异范数: {torch.norm(output - x).item():.4f}")

