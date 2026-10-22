import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class GuidelineAlignment(nn.Module):
    """
    解释一致性修正
    方法：
    1. 硬约束：强制某些边必须存在/不存在
    2. 软约束：通过损失函数鼓励某些边的存在/强度
    3. 语义对齐：确保边的语义方向与Guidelines一致
    """
    
    def __init__(
        self, 
        alignment_mode='soft',
        constraint_weight=1.0,
        temperature=1.0
    ):
        """
        Args:
            alignment_mode: 对齐模式 ('hard', 'soft', 'semantic')
            constraint_weight: 约束权重
            temperature: 软约束的温度参数
        """
        super().__init__()
        self.alignment_mode = alignment_mode
        self.constraint_weight = constraint_weight
        self.temperature = temperature
    
    def forward(
        self,
        stats_edge_matrix: torch.Tensor,
        explain_edge_matrix: Optional[torch.Tensor],
        node_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算对齐后的边权矩阵和对齐损失
        Args:
            stats_edge_matrix: 统计DAG的边权矩阵 [N, N]
            explain_edge_matrix: 解释性DAG的边权矩阵 [N, N], 可选
            node_embeddings: 节点嵌入 [N, D], 用于语义对齐
            
        Returns:
            aligned_matrix: 对齐后的边权矩阵
            alignment_loss: 对齐损失
        """
        if explain_edge_matrix is None:
            # 没有解释性DAG，返回原始统计DAG
            return stats_edge_matrix, torch.tensor(0.0, device=stats_edge_matrix.device)
        
        if self.alignment_mode == 'hard':
            return self._hard_alignment(stats_edge_matrix, explain_edge_matrix)
        elif self.alignment_mode == 'soft':
            return self._soft_alignment(stats_edge_matrix, explain_edge_matrix)
        elif self.alignment_mode == 'semantic':
            if node_embeddings is None:
                # 回退到软约束
                return self._soft_alignment(stats_edge_matrix, explain_edge_matrix)
            return self._semantic_alignment(stats_edge_matrix, explain_edge_matrix, node_embeddings)
        else:
            raise ValueError(f"Unknown alignment mode: {self.alignment_mode}")
    
    def _hard_alignment(
        self,
        stats_matrix: torch.Tensor,
        explain_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬约束对齐：强制统计DAG遵循解释性DAG的结构
        规则：
        1. 如果解释性DAG中存在边(i→j)，则保留统计DAG中的对应边
        2. 如果解释性DAG中不存在边(i→j)，则删除统计DAG中的对应边
        """
        # 创建二值掩码：解释性DAG中存在边的位置
        explain_mask = (explain_matrix > 0).float()
        
        # 应用掩码
        aligned_matrix = stats_matrix * explain_mask
        
        # 硬约束没有损失（已经强制对齐）
        loss = torch.tensor(0.0, device=stats_matrix.device)
        
        return aligned_matrix, loss
    
    def _soft_alignment(
        self,
        stats_matrix: torch.Tensor,
        explain_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软约束对齐：通过损失函数鼓励统计DAG与解释性DAG一致
        
        损失函数：L_align = Σ_ij |W_stats[i,j] - W_explain[i,j]|²
        
        但不直接修改统计DAG，而是返回加权平均
        """
        # 计算对齐损失（MSE）
        loss = F.mse_loss(stats_matrix, explain_matrix, reduction='mean')
        loss = self.constraint_weight * loss
        
        # 加权平均融合
        alpha = torch.sigmoid(torch.tensor(self.temperature))
        aligned_matrix = alpha * explain_matrix + (1 - alpha) * stats_matrix
        
        return aligned_matrix, loss
    
    def _semantic_alignment(
        self,
        stats_matrix: torch.Tensor,
        explain_matrix: torch.Tensor,
        node_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        语义对齐：基于节点语义相似度调整边权
        思路：
        1. 如果两个节点语义相似度高，则更倾向于保留统计DAG的边
        2. 如果语义相似度低，则更倾向于遵循解释性DAG
        """
        # 计算节点间的语义相似度矩阵
        node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
        semantic_sim = torch.mm(node_embeddings, node_embeddings.t())
        semantic_sim = torch.clamp(semantic_sim, 0, 1)  # [0, 1]
        
        # 语义相似度作为权重
        # 高相似度 → 更信任统计DAG
        # 低相似度 → 更信任解释性DAG
        alpha = semantic_sim
        aligned_matrix = alpha * stats_matrix + (1 - alpha) * explain_matrix
        
        # 计算对齐损失：鼓励语义一致性
        # 如果解释性DAG有边但语义相似度低，惩罚
        explain_mask = (explain_matrix > 0).float()
        semantic_inconsistency = explain_mask * (1 - semantic_sim)
        loss = self.constraint_weight * semantic_inconsistency.mean()
        
        return aligned_matrix, loss
    
    def compute_structure_consistency_loss(
        self,
        edge_matrix: torch.Tensor,
        explain_edges: Optional[List[Tuple[int, int]]] = None
    ) -> torch.Tensor:
        """
        计算结构一致性损失
        目标：
        1. 有向无环 (DAG property)
        2. 稀疏性 (Sparsity)
        3. 与解释性边一致 (Guideline consistency)
        
        Args:
            edge_matrix: 边权矩阵 [N, N]
            explain_edges: 解释性DAG的边列表 [(src, dst), ...]
            
        Returns:
            consistency_loss: 结构一致性损失
        """
        device = edge_matrix.device
        N = edge_matrix.size(0)
        
        # 1. DAG约束：检测环（通过矩阵幂）
        # 如果存在环，则 Tr((I + A)^N) > N
        A = torch.clamp(edge_matrix, 0, 1)  # 二值化
        A_power = A
        trace_sum = torch.trace(A_power)
        
        for _ in range(min(N - 1, 10)):  # 限制迭代次数避免OOM
            A_power = torch.mm(A_power, A)
            trace_sum += torch.trace(A_power)
        
        # 如果有环，trace_sum会很大
        dag_loss = F.relu(trace_sum - N)
        
        # 2. 稀疏性约束：L1正则
        sparsity_loss = torch.mean(torch.abs(edge_matrix))
        
        # 3. 解释性一致性
        guideline_loss = torch.tensor(0.0, device=device)
        if explain_edges is not None and len(explain_edges) > 0:
            # 鼓励解释性边具有较高权重
            for src, dst in explain_edges:
                if src < N and dst < N:
                    # 最小化 -log(edge_weight)，即鼓励高权重
                    guideline_loss += -torch.log(edge_matrix[src, dst] + 1e-8)
            guideline_loss /= len(explain_edges)
        
        # 总损失
        total_loss = 0.1 * dag_loss + 0.01 * sparsity_loss + 0.5 * guideline_loss
        
        return total_loss


class GuidelineEncoder(nn.Module):
    def __init__(self, encoder, tokenizer):
        """
        Args:
            encoder: BERT encoder
            tokenizer: BERT tokenizer
        """
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
    
    def encode_guidelines(
        self,
        guidelines: List[str],
        device: torch.device
    ) -> torch.Tensor:
        """
        编码Guidelines文本
        
        Args:
            guidelines: Guidelines文本列表
            device: 计算设备
            
        Returns:
            guideline_embeddings: [num_guidelines, hidden_dim]
        """
        if len(guidelines) == 0:
            return None
        
        # Tokenize
        inputs = self.tokenizer(
            guidelines,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        
        return embeddings
    
    def compute_guideline_similarity(
        self,
        node_embeddings: torch.Tensor,
        guideline_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        计算节点与Guidelines的相似度
        
        Args:
            node_embeddings: [N, D]
            guideline_embeddings: [M, D]
            
        Returns:
            similarity_matrix: [N, M]
        """
        node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
        guideline_embeddings = F.normalize(guideline_embeddings, p=2, dim=-1)
        
        similarity = torch.mm(node_embeddings, guideline_embeddings.t())
        return similarity


def load_task_guidelines(task_name: str, guidelines_file: str) -> Optional[List[str]]:
    """
    Args:
        task_name: 任务名称
        guidelines_file: Guidelines文件路径
        
    Returns:
        guidelines: Guidelines文本列表
    """
    import json
    import os
    
    if not os.path.exists(guidelines_file):
        return None
    
    try:
        with open(guidelines_file, 'r', encoding='utf-8') as f:
            all_guidelines = json.load(f)
        
        if task_name in all_guidelines:
            return all_guidelines[task_name]
        else:
            return None
    except Exception as e:
        print(f"加载Guidelines失败: {e}")
        return None

