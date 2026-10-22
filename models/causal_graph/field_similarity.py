import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class FieldLevelSimilarity(nn.Module):
    """
    字段级相似度计算模块
    1. 字段化编码: 对每个字段单独编码
    2. 类内原型: 计算每个标签在每个字段上的原型
    3. 字段重要性: 计算或学习字段权重
    4. 加权相似度: 融合字段权重和语义相似度
    """
    
    def __init__(
        self,
        field_weight_type='uniform',  # 'uniform', 'info_gain', 'learnable'
        hidden_dim=768,
        num_fields=None,
        normalize=True
    ):
        """
        Args:
            field_weight_type: 字段权重类型
                - 'uniform': 均一权重 γ_k = 1/K
                - 'info_gain': 基于信息增益
                - 'learnable': 可学习的注意力权重
            hidden_dim: 编码器隐藏层维度
            num_fields: 字段数量（如果使用learnable）
            normalize: 是否归一化字段权重
        """
        super().__init__()
        self.field_weight_type = field_weight_type
        self.hidden_dim = hidden_dim
        self.normalize = normalize
        
        # 如果使用可学习权重，初始化注意力层
        if field_weight_type == 'learnable' and num_fields is not None:
            self.weight_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        else:
            self.weight_attention = None
    
    def encode_field(
        self,
        field_name: str,
        field_value: str,
        tokenizer,
        encoder,
        device
    ) -> torch.Tensor:
        """
        字段化编码
        
        公式: f_k(x) = "{name}_k: {value}_k"
              u_k(x) = Encoder(f_k(x))
        
        Args:
            field_name: 字段名称
            field_value: 字段取值
            tokenizer: BERT tokenizer
            encoder: BERT encoder
            device: 计算设备
            
        Returns:
            字段嵌入向量 [hidden_dim]
        """
        # 构造字段模板
        field_text = f"{field_name}: {field_value}"
        
        # 编码
        inputs = tokenizer(
            field_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = encoder(**inputs)
            # 使用CLS token表示或平均池化
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                field_emb = outputs.pooler_output.squeeze(0)
            else:
                field_emb = outputs.last_hidden_state[:, 0].squeeze(0)
        
        return field_emb
    
    def compute_label_prototypes(
        self,
        samples: List[Dict],
        field_names: List[str],
        tokenizer,
        encoder,
        device
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        计算类内原型 (Label-specific Prototype)
        Args:
            samples: 样本列表，每个样本包含字段和标签
            field_names: 字段名称列表
            tokenizer: BERT tokenizer
            encoder: BERT encoder
            device: 计算设备
            
        Returns:
            prototypes: {label: {field_name: tensor}}
        """
        # 按标签分组样本
        label_samples = defaultdict(list)
        for sample in samples:
            label = sample.get('lbl', None)
            if label is not None:
                label_samples[label].append(sample)
        
        # 计算每个标签在每个字段上的原型
        prototypes = {}
        for label, label_sample_list in label_samples.items():
            field_prototypes = {}
            for field_name in field_names:
                # 收集该字段的所有取值嵌入
                field_embeddings = []
                for sample in label_sample_list:
                    field_value = sample.get(field_name, None)
                    if field_value is not None:
                        field_emb = self.encode_field(
                            field_name, str(field_value),
                            tokenizer, encoder, device
                        )
                        field_embeddings.append(field_emb)
                
                # 计算平均原型
                if len(field_embeddings) > 0:
                    field_prototype = torch.stack(field_embeddings).mean(dim=0)
                    field_prototypes[field_name] = field_prototype
            
            prototypes[label] = field_prototypes
        
        return prototypes
    
    def compute_field_weights_uniform(
        self,
        num_fields: int
    ) -> np.ndarray:
        """
        均一权重
        
        公式: γ_k = 1/K
        
        Args:
            num_fields: 字段数量
            
        Returns:
            字段权重数组 [num_fields]
        """
        return np.ones(num_fields) / num_fields
    
    def compute_field_weights_info_gain(
        self,
        samples: List[Dict],
        field_names: List[str]
    ) -> np.ndarray:
        """
        基于信息增益的字段权重
        Args:
            samples: 样本列表
            field_names: 字段名称列表
            
        Returns:
            字段权重数组 [num_fields]
        """
        # 计算每个字段的信息增益
        info_gains = []
        labels = [s['lbl'] for s in samples if 'lbl' in s]
        
        for field_name in field_names:
            # 计算信息增益（简化版：使用类别熵）
            field_values = [str(s.get(field_name, 'N/A')) for s in samples]
            
            # H(Y) - H(Y|X)
            ig = self._compute_mutual_information(labels, field_values)
            info_gains.append(ig)
        
        # 归一化
        info_gains = np.array(info_gains)
        if info_gains.sum() > 0:
            weights = info_gains / info_gains.sum()
        else:
            weights = np.ones(len(field_names)) / len(field_names)
        
        return weights
    
    def _compute_mutual_information(
        self,
        labels: List[str],
        values: List[str]
    ) -> float:
        """
        计算互信息 I(Y; X)
        
        简化实现: I(Y;X) = H(Y) - H(Y|X)
        """
        from collections import Counter
        
        n = len(labels)
        if n == 0:
            return 0.0
        
        # H(Y)
        label_counts = Counter(labels)
        h_y = -sum((c/n) * np.log2(c/n) for c in label_counts.values())
        
        # H(Y|X)
        value_groups = defaultdict(list)
        for label, value in zip(labels, values):
            value_groups[value].append(label)
        
        h_y_given_x = 0.0
        for value, group_labels in value_groups.items():
            p_x = len(group_labels) / n
            group_counts = Counter(group_labels)
            h_y_x = -sum((c/len(group_labels)) * np.log2(c/len(group_labels)) 
                        for c in group_counts.values())
            h_y_given_x += p_x * h_y_x
        
        return h_y - h_y_given_x
    
    def compute_field_weights_learnable(
        self,
        field_embeddings: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        可学习的注意力权重
        
        公式: γ_k = softmax(W_γ u_k(x))
        
        Args:
            field_embeddings: 字段嵌入列表 [num_fields, hidden_dim]
            
        Returns:
            字段权重 [num_fields]
        """
        if self.weight_attention is None:
            raise ValueError("weight_attention not initialized")
        
        field_embs = torch.stack(field_embeddings)  # [num_fields, hidden_dim]
        logits = self.weight_attention(field_embs).squeeze(-1)  # [num_fields]
        weights = F.softmax(logits, dim=0)
        
        return weights
    
    def compute_sample_label_similarity(
        self,
        sample: Dict,
        label: str,
        field_names: List[str],
        label_prototypes: Dict[str, Dict[str, torch.Tensor]],
        field_weights: np.ndarray,
        tokenizer,
        encoder,
        device
    ) -> float:
        """
        样本-标签字段加权相似度
        Args:
            sample: 输入样本
            label: 目标标签
            field_names: 字段名称列表
            label_prototypes: 标签原型 {label: {field: tensor}}
            field_weights: 字段权重数组
            tokenizer: BERT tokenizer
            encoder: BERT encoder
            device: 计算设备
            
        Returns:
            加权相似度分数
        """
        if label not in label_prototypes:
            return 0.0
        
        total_similarity = 0.0
        for i, field_name in enumerate(field_names):
            field_value = sample.get(field_name, None)
            if field_value is None or field_name not in label_prototypes[label]:
                continue
            
            # 编码样本字段
            field_emb = self.encode_field(
                field_name, str(field_value),
                tokenizer, encoder, device
            )
            
            # 获取标签原型
            prototype_emb = label_prototypes[label][field_name]
            
            # 计算余弦相似度
            cos_sim = F.cosine_similarity(
                field_emb.unsqueeze(0),
                prototype_emb.unsqueeze(0)
            ).item()
            
            # 加权
            total_similarity += field_weights[i] * cos_sim
        
        return total_similarity


def compute_task_edge_weights_field_level(
    task_name: str,
    task_data: Dict,
    field_names: List[str],
    tokenizer,
    encoder,
    device,
    field_weight_type='uniform',
    edge_keep_ratio=0.3
) -> Dict:
    """
    使用字段级相似度计算任务的边权
    公式: w_{i→j}^{(T)} = (1/|X_i|) Σ_{x∈X_i} sim(x,j)
          其中 sim(x,j) = Σ_k γ_k · cos(u_k(x), p_{j,k})
    
    Args:
        task_name: 任务名称
        task_data: 任务数据
        field_names: 字段名称列表
        tokenizer: BERT tokenizer
        encoder: BERT encoder
        device: 计算设备
        field_weight_type: 字段权重类型
        edge_keep_ratio: 保留边的比例
        
    Returns:
        DAG字典 {nodes, edges, labels, label_nodes}
    """
    # 初始化字段级相似度模块
    field_sim = FieldLevelSimilarity(
        field_weight_type=field_weight_type,
        hidden_dim=768
    )
    
    # 获取标签和样本
    labels = list(task_data.get('target_key', []))
    train_samples = task_data.get('train', [])
    
    if len(labels) == 0 or len(train_samples) == 0:
        return {
            "nodes": [],
            "edges": [],
            "labels": [],
            "label_nodes": {}
        }
    
    # 1. 计算类内原型
    print(f"  Computing label prototypes for {len(labels)} labels...")
    label_prototypes = field_sim.compute_label_prototypes(
        train_samples, field_names, tokenizer, encoder, device
    )
    
    # 2. 计算字段权重
    print(f"  Computing field weights ({field_weight_type})...")
    if field_weight_type == 'uniform':
        field_weights = field_sim.compute_field_weights_uniform(len(field_names))
    elif field_weight_type == 'info_gain':
        field_weights = field_sim.compute_field_weights_info_gain(
            train_samples, field_names
        )
    else:
        # 默认均一
        field_weights = field_sim.compute_field_weights_uniform(len(field_names))
    
    print(f"  Field weights: {dict(zip(field_names, field_weights))}")
    
    # 3. 计算边权矩阵 (label-to-label)
    print(f"  Computing edge weights using field-level similarity...")
    edge_weights = np.zeros((len(labels), len(labels)))
    
    for i, label_i in enumerate(labels):
        # 获取该标签的所有样本
        label_i_samples = [s for s in train_samples if s.get('lbl') == label_i]
        
        if len(label_i_samples) == 0:
            continue
        
        for j, label_j in enumerate(labels):
            # 计算样本-标签相似度的平均
            similarities = []
            for sample in label_i_samples[:min(100, len(label_i_samples))]:  # 限制样本数
                sim = field_sim.compute_sample_label_similarity(
                    sample, label_j, field_names, label_prototypes,
                    field_weights, tokenizer, encoder, device
                )
                similarities.append(sim)
            
            if len(similarities) > 0:
                edge_weights[i, j] = np.mean(similarities)
    
    # 4. 构建DAG节点和边
    nodes = [{"id": i, "name": str(lbl), "text": f"label: {lbl}"} 
             for i, lbl in enumerate(labels)]
    
    # 5. 剪枝: 保留Top-K边
    edges = []
    np.fill_diagonal(edge_weights, -np.inf)  # 不要自环
    
    k = max(1, int(np.ceil(len(labels) * edge_keep_ratio)))
    for i in range(len(labels)):
        idx = np.argsort(edge_weights[i])[::-1][:k]
        vals = edge_weights[i][idx]
        for j, v in zip(idx.tolist(), vals.tolist()):
            if np.isfinite(v) and v > 0:
                edges.append({"src": int(i), "dst": int(j), "w": float(v)})
    
    # 6. 标签节点一对一映射
    label_nodes = {str(lbl): [i] for i, lbl in enumerate(labels)}
    
    print(f"  Built DAG: {len(nodes)} nodes, {len(edges)} edges")
    
    return {
        "nodes": nodes,
        "edges": edges,
        "labels": labels,
        "label_nodes": label_nodes
    }

