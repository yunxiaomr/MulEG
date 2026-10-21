"""
符号编码器 (Symbolic Encoder)
将符号任务转化为图结构编码，支持规则推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Tuple, Optional
from torch_geometric.nn import SAGEConv, global_mean_pool


class SymbolicGraphEncoder(nn.Module):
    """
    将符号任务转化为图结构编码
    例如: 井字棋 → 9节点图（边表示相邻位置）
    """
    
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # GNN编码器 (GraphSAGE)
        self.conv1 = SAGEConv(embed_dim, embed_dim // 2)
        self.conv2 = SAGEConv(embed_dim // 2, embed_dim)
        
        # 位置编码
        self.position_emb = nn.Embedding(9, embed_dim)  # 最多9个位置
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim // 2)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, node_features: torch.Tensor, 
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            node_features: (num_nodes, embed_dim) 节点特征
            edge_index: (2, num_edges) 边索引
            batch: (num_nodes,) batch索引，用于区分不同样本的节点
        
        Returns:
            graph_emb: (batch_size, embed_dim) 或 (embed_dim,) 图级别嵌入
        """
        # 1. 添加位置编码
        num_nodes = node_features.size(0)
        pos = torch.arange(num_nodes, device=node_features.device) % 9  # 循环位置
        pos_emb = self.position_emb(pos)
        x = node_features + pos_emb
        
        # 2. GNN编码
        x = F.relu(self.norm1(self.conv1(x, edge_index)))
        x = self.norm2(self.conv2(x, edge_index))
        
        # 3. 图级别池化
        if batch is not None:
            # 多图batch，使用global_mean_pool
            graph_emb = global_mean_pool(x, batch)  # (batch_size, embed_dim)
        else:
            # 单图，直接平均
            graph_emb = torch.mean(x, dim=0)  # (embed_dim,)
        
        return graph_emb


def encode_tic_tac_toe_as_graph(sample_text: str, 
                                 bert_encoder, 
                                 tokenizer,
                                 device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将井字棋样本转化为图结构
    
    Args:
        sample_text: "top-left is x, top-middle is x, ..."
        bert_encoder: BERT编码器
        tokenizer: BERT tokenizer
        device: 设备
    
    Returns:
        node_features: (9, 768) - 9个位置的BERT编码
        edge_index: (2, E) - 相邻关系边
    """
    # 解析位置状态
    positions = [
        'top-left', 'top-middle', 'top-right',
        'middle-left', 'middle-middle', 'middle-right',
        'bottom-left', 'bottom-middle', 'bottom-right'
    ]
    
    # 提取每个位置的状态 (x, o, b)
    states = {}
    for pos in positions:
        # 从sample_text中提取 "{pos} is {state}"
        match = re.search(f"{pos} is (\\w+)", sample_text)
        if match:
            states[pos] = match.group(1)
        else:
            states[pos] = 'b'  # 空白
    
    # BERT编码每个位置
    node_texts = [f"{pos} is {states[pos]}" for pos in positions]
    inputs = tokenizer(node_texts, padding=True, return_tensors='pt', truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = bert_encoder(**inputs)
        node_features = outputs.pooler_output  # (9, 768)
    
    # 构建边（相邻关系 + 同行/列/对角线）
    edges = [
        # 水平相邻
        (0,1), (1,2), (3,4), (4,5), (6,7), (7,8),
        # 垂直相邻
        (0,3), (3,6), (1,4), (4,7), (2,5), (5,8),
        # 对角线
        (0,4), (4,8), (2,4), (4,6)
    ]
    
    # 双向边
    edge_index = []
    for (u, v) in edges:
        edge_index.append([u, v])
        edge_index.append([v, u])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()  # (2, E)
    
    return node_features, edge_index


def add_rule_prompt(sample_text: str, task_name: str) -> str:
    """
    在输入中加入隐式规则提示
    
    Args:
        sample_text: 原始样本文本
        task_name: 任务名称
    
    Returns:
        enhanced_text: 添加规则提示后的文本
    """
    rule_templates = {
        'tic-tac-toe-endgame': "Rule: Three X or O in a row wins. ",
        'driving-championship-points': "Rule: Points are calculated as: 25×wins + 18×podiums + 1×poles. ",
        'manifold-orientability': "Rule: A manifold is orientable if it has a consistent normal direction. ",
        'proteinogenic-acid': "Rule: There are 20 standard amino acids in proteins. "
    }
    
    rule = rule_templates.get(task_name, "")
    enhanced_text = rule + sample_text
    
    return enhanced_text


def is_symbolic_task(task_name: str) -> bool:
    """检查是否为符号/规则任务"""
    symbolic_tasks = [
        'tic-tac-toe-endgame',
        'driving-championship-points',
        'manifold-orientability',
        'proteinogenic-acid'
    ]
    return task_name in symbolic_tasks


class HybridEncoderWithSymbolic(nn.Module):
    """
    扩展的混合编码器：文本 + 数值 + 符号图
    """
    
    def __init__(self, bert_encoder, embed_dim=768):
        super().__init__()
        self.bert = bert_encoder
        self.embed_dim = embed_dim
        
        # 导入数值编码器
        from .hybrid_encoder import NumericalFeatureEncoder
        
        self.numeric_encoder = NumericalFeatureEncoder(output_dim=embed_dim)
        self.symbolic_encoder = SymbolicGraphEncoder(embed_dim=embed_dim)

        # 三路融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def set_view_mode(self, mode: str):
        """设置数值特征编码的视图模式 (用于消融实验)."""
        self.numeric_encoder.set_view_mode(mode)

    def forward(self, text_input: Dict,
                numeric_features: Optional[torch.Tensor] = None,
                symbolic_graph: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            text_input: BERT输入字典
            numeric_features: (batch_size, num_features) 数值特征
            symbolic_graph: 符号图 {'node_features': ..., 'edge_index': ...}
        
        Returns:
            fused_emb: (batch_size, embed_dim) 融合后的嵌入
        """
        # 1. BERT编码
        bert_outputs = self.bert(**text_input)
        text_emb = bert_outputs.pooler_output  # (B, D)
        batch_size = text_emb.size(0)
        device = text_emb.device
        
        # 2. 数值编码
        if numeric_features is not None and numeric_features.size(1) > 0:
            numeric_emb = self.numeric_encoder(numeric_features)  # (B, D)
        else:
            numeric_emb = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # 3. 符号图编码（如果有）
        if symbolic_graph is not None:
            symbolic_emb = self.symbolic_encoder(
                symbolic_graph['node_features'],
                symbolic_graph['edge_index'],
                symbolic_graph.get('batch', None)
            )
            # 确保形状一致
            if symbolic_emb.dim() == 1:
                symbolic_emb = symbolic_emb.unsqueeze(0).expand(batch_size, -1)
            elif symbolic_emb.size(0) != batch_size:
                # 重复或裁剪以匹配batch_size
                if symbolic_emb.size(0) == 1:
                    symbolic_emb = symbolic_emb.expand(batch_size, -1)
                else:
                    symbolic_emb = symbolic_emb[:batch_size]
        else:
            symbolic_emb = torch.zeros(batch_size, self.embed_dim, device=device)
        
        # 4. 融合
        combined = torch.cat([text_emb, numeric_emb, symbolic_emb], dim=-1)  # (B, 3D)
        fused_emb = self.fusion(combined)  # (B, D)
        
        return fused_emb


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("符号编码器测试")
    print("=" * 60)
    
    # 测试规则提示
    test_sample = "top-left is x, top-middle is x, top-right is o"
    task_name = "tic-tac-toe-endgame"
    
    enhanced = add_rule_prompt(test_sample, task_name)
    print(f"\n原始文本:\n  {test_sample}")
    print(f"\n添加规则后:\n  {enhanced}")
    
    # 测试符号图编码器
    print("\n" + "=" * 60)
    print("SymbolicGraphEncoder测试")
    print("=" * 60)
    
    embed_dim = 768
    num_nodes = 9
    num_edges = 20
    
    encoder = SymbolicGraphEncoder(embed_dim=embed_dim)
    
    # 模拟数据
    node_features = torch.randn(num_nodes, embed_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # 单图编码
    graph_emb = encoder(node_features, edge_index)
    print(f"\n单图编码:")
    print(f"  输入节点特征: {node_features.shape}")
    print(f"  边索引: {edge_index.shape}")
    print(f"  图嵌入: {graph_emb.shape}")
    
    # 多图batch编码
    batch_size = 4
    total_nodes = num_nodes * batch_size
    node_features_batch = torch.randn(total_nodes, embed_dim)
    edge_index_batch = torch.cat([
        edge_index + i * num_nodes for i in range(batch_size)
    ], dim=1)
    batch_idx = torch.repeat_interleave(torch.arange(batch_size), num_nodes)
    
    graph_emb_batch = encoder(node_features_batch, edge_index_batch, batch_idx)
    print(f"\n多图batch编码:")
    print(f"  输入节点特征: {node_features_batch.shape}")
    print(f"  边索引: {edge_index_batch.shape}")
    print(f"  Batch索引: {batch_idx.shape}")
    print(f"  图嵌入: {graph_emb_batch.shape}")

