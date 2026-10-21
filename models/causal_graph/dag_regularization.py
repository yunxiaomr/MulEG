"""
DAG正则化损失模块
提供多种DAG结构约束和正则化方法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple


class DAGRegularization(nn.Module):
    """
    DAG正则化损失
    
    包含以下约束：
    1. DAG约束：确保有向无环图性质
    2. 稀疏性约束：鼓励稀疏的边连接
    3. 平滑性约束：相邻节点的边权应相似
    4. 解释性一致性：与人工定义的DAG结构一致
    5. 因果GNN正则：节点嵌入的因果一致性
    """
    
    def __init__(
        self,
        lambda_dag: float = 0.1,
        lambda_sparsity: float = 0.01,
        lambda_smooth: float = 0.05,
        lambda_explain: float = 0.5,
        lambda_causal: float = 0.1,
        dag_mode: str = 'trace'  # 'trace' or 'polynomial'
    ):
        """
        Args:
            lambda_dag: DAG约束权重
            lambda_sparsity: 稀疏性约束权重
            lambda_smooth: 平滑性约束权重
            lambda_explain: 解释性一致性权重
            lambda_causal: 因果正则权重
            dag_mode: DAG约束模式
        """
        super().__init__()
        self.lambda_dag = lambda_dag
        self.lambda_sparsity = lambda_sparsity
        self.lambda_smooth = lambda_smooth
        self.lambda_explain = lambda_explain
        self.lambda_causal = lambda_causal
        self.dag_mode = dag_mode
    
    def forward(
        self,
        edge_weights: torch.Tensor,
        explain_edges: Optional[torch.Tensor] = None,
        node_embeddings: Optional[torch.Tensor] = None,
        adjacency_matrix: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有正则化损失
        
        Args:
            edge_weights: 边权矩阵 [N, N]
            explain_edges: 解释性DAG的边权矩阵 [N, N]
            node_embeddings: 节点嵌入 [N, D]
            adjacency_matrix: 邻接矩阵（用于平滑性）[N, N]
            
        Returns:
            losses: 各项损失的字典
        """
        losses = {}
        device = edge_weights.device
        
        # 1. DAG约束
        if self.lambda_dag > 0:
            dag_loss = self.compute_dag_constraint(edge_weights)
            losses['dag'] = self.lambda_dag * dag_loss
        else:
            losses['dag'] = torch.tensor(0.0, device=device)
        
        # 2. 稀疏性约束
        if self.lambda_sparsity > 0:
            sparsity_loss = self.compute_sparsity_loss(edge_weights)
            losses['sparsity'] = self.lambda_sparsity * sparsity_loss
        else:
            losses['sparsity'] = torch.tensor(0.0, device=device)
        
        # 3. 平滑性约束
        if self.lambda_smooth > 0 and adjacency_matrix is not None:
            smooth_loss = self.compute_smoothness_loss(edge_weights, adjacency_matrix)
            losses['smoothness'] = self.lambda_smooth * smooth_loss
        else:
            losses['smoothness'] = torch.tensor(0.0, device=device)
        
        # 4. 解释性一致性
        if self.lambda_explain > 0 and explain_edges is not None:
            explain_loss = self.compute_explanation_consistency(edge_weights, explain_edges)
            losses['explanation'] = self.lambda_explain * explain_loss
        else:
            losses['explanation'] = torch.tensor(0.0, device=device)
        
        # 5. 因果GNN正则
        if self.lambda_causal > 0 and node_embeddings is not None:
            causal_loss = self.compute_causal_regularization(node_embeddings, edge_weights)
            losses['causal'] = self.lambda_causal * causal_loss
        else:
            losses['causal'] = torch.tensor(0.0, device=device)
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_dag_constraint(self, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        计算DAG约束损失
        
        目标：确保图是有向无环图
        
        方法1 (trace): h(W) = tr(e^{W ⊙ W}) - d
        方法2 (polynomial): h(W) = tr((I + |W|)^d) - d
        
        如果h(W) > 0，则存在环
        """
        N = edge_weights.size(0)
        
        if self.dag_mode == 'trace':
            # 矩阵指数方法（更精确但计算量大）
            W_squared = edge_weights * edge_weights
            # 使用泰勒展开近似 e^W
            # e^W ≈ I + W + W^2/2 + W^3/6 + ...
            M = torch.eye(N, device=edge_weights.device)
            exp_W = M.clone()
            M_k = M.clone()
            
            for k in range(1, 6):  # 前5项
                M_k = torch.mm(M_k, W_squared) / k
                exp_W = exp_W + M_k
            
            h = torch.trace(exp_W) - N
            dag_loss = F.relu(h)
            
        elif self.dag_mode == 'polynomial':
            # 多项式方法（更快但不太精确）
            W_abs = torch.abs(edge_weights)
            I = torch.eye(N, device=edge_weights.device)
            M = I + W_abs
            
            # 计算 M^N 的迹
            M_power = M
            trace_sum = torch.trace(M_power)
            
            for _ in range(min(N - 1, 10)):
                M_power = torch.mm(M_power, M)
                trace_sum += torch.trace(M_power)
            
            h = trace_sum - N
            dag_loss = F.relu(h)
        else:
            raise ValueError(f"Unknown DAG mode: {self.dag_mode}")
        
        return dag_loss
    
    def compute_sparsity_loss(self, edge_weights: torch.Tensor) -> torch.Tensor:
        """
        计算稀疏性损失
        
        目标：鼓励大部分边权接近0
        
        方法：L1正则 + 熵正则
        """
        # L1正则
        l1_loss = torch.mean(torch.abs(edge_weights))
        
        # 熵正则：鼓励边权分布集中
        # H(p) = -Σ p_i log(p_i)
        # 我们希望最小化熵，即鼓励少数边有高权重
        edge_probs = F.softmax(edge_weights.flatten(), dim=0)
        entropy = -torch.sum(edge_probs * torch.log(edge_probs + 1e-10))
        
        # 组合损失
        sparsity_loss = l1_loss - 0.1 * entropy  # 负号：最小化熵
        
        return sparsity_loss
    
    def compute_smoothness_loss(
        self,
        edge_weights: torch.Tensor,
        adjacency_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        计算平滑性损失
        
        目标：相邻节点的出边权重应相似
        
        方法：L_smooth = Σ_{i~j} ||W[i,:] - W[j,:]||²
        """
        N = edge_weights.size(0)
        smooth_loss = torch.tensor(0.0, device=edge_weights.device)
        
        # 遍历相邻节点对
        adj_indices = torch.nonzero(adjacency_matrix > 0, as_tuple=False)
        
        if len(adj_indices) > 0:
            for idx in adj_indices:
                i, j = idx[0].item(), idx[1].item()
                diff = edge_weights[i, :] - edge_weights[j, :]
                smooth_loss += torch.sum(diff ** 2)
            
            smooth_loss /= len(adj_indices)
        
        return smooth_loss
    
    def compute_explanation_consistency(
        self,
        edge_weights: torch.Tensor,
        explain_edges: torch.Tensor
    ) -> torch.Tensor:
        """
        计算解释性一致性损失
        
        目标：统计DAG的边权应与解释性DAG一致
        
        方法：
        1. 对于解释性DAG中存在的边，鼓励高权重
        2. 对于解释性DAG中不存在的边，鼓励低权重
        """
        # 二值掩码
        explain_mask = (explain_edges > 0).float()
        
        # 存在的边：最小化 -log(W)
        exist_loss = -torch.sum(explain_mask * torch.log(edge_weights + 1e-8))
        exist_loss /= (torch.sum(explain_mask) + 1e-8)
        
        # 不存在的边：最小化 log(1 - W)
        non_exist_mask = 1.0 - explain_mask
        non_exist_loss = -torch.sum(non_exist_mask * torch.log(1 - edge_weights + 1e-8))
        non_exist_loss /= (torch.sum(non_exist_mask) + 1e-8)
        
        # 组合
        consistency_loss = exist_loss + 0.5 * non_exist_loss
        
        return consistency_loss
    
    def compute_causal_regularization(
        self,
        node_embeddings: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        计算因果GNN正则损失
        
        目标：节点嵌入应反映因果结构
        
        方法：
        1. 因果一致性：如果有边i→j，则h_i和h_j应相似
        2. 因果方向性：h_i应包含预测h_j的信息
        3. 独立性：无边连接的节点应独立
        """
        N = node_embeddings.size(0)
        device = node_embeddings.device
        
        # 归一化嵌入
        node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
        
        # 计算节点相似度矩阵
        similarity = torch.mm(node_embeddings, node_embeddings.t())
        
        # 1. 因果一致性：有边的节点应相似
        edge_mask = (edge_weights > 0.1).float()  # 阈值化
        consistency_loss = -torch.sum(edge_mask * similarity)
        consistency_loss /= (torch.sum(edge_mask) + 1e-8)
        
        # 2. 独立性：无边的节点应不相似
        non_edge_mask = 1.0 - edge_mask
        independence_loss = torch.sum(non_edge_mask * torch.abs(similarity))
        independence_loss /= (torch.sum(non_edge_mask) + 1e-8)
        
        # 3. 因果方向性：通过HSIC（Hilbert-Schmidt Independence Criterion）
        # 简化版：如果有边i→j，则Cov(h_i, h_j) > 0
        directional_loss = torch.tensor(0.0, device=device)
        
        edge_indices = torch.nonzero(edge_mask > 0, as_tuple=False)
        if len(edge_indices) > 0:
            for idx in edge_indices[:100]:  # 限制计算量
                i, j = idx[0].item(), idx[1].item()
                h_i = node_embeddings[i]
                h_j = node_embeddings[j]
                
                # 计算协方差（简化）
                cov = torch.dot(h_i, h_j)
                directional_loss += F.relu(-cov)  # 惩罚负协方差
            
            directional_loss /= len(edge_indices[:100])
        
        # 总损失
        causal_loss = consistency_loss + 0.3 * independence_loss + 0.2 * directional_loss
        
        return causal_loss


class CausalGNNRegularizer(nn.Module):
    """
    因果GNN专用正则化器
    
    基于NEW_method.md中的"因果GNN正则"部分
    """
    
    def __init__(
        self,
        lambda_intervene: float = 0.1,
        lambda_counterfactual: float = 0.05
    ):
        """
        Args:
            lambda_intervene: 干预一致性权重
            lambda_counterfactual: 反事实一致性权重
        """
        super().__init__()
        self.lambda_intervene = lambda_intervene
        self.lambda_counterfactual = lambda_counterfactual
    
    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_weights: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算因果GNN正则损失
        
        Args:
            node_embeddings: GNN输出的节点嵌入 [batch, N, D]
            edge_weights: 边权矩阵 [N, N]
            labels: 标签 [batch]
            
        Returns:
            losses: 损失字典
        """
        losses = {}
        device = node_embeddings.device
        
        # 1. 干预一致性损失
        if self.lambda_intervene > 0:
            intervene_loss = self.compute_intervention_consistency(
                node_embeddings, edge_weights
            )
            losses['intervention'] = self.lambda_intervene * intervene_loss
        else:
            losses['intervention'] = torch.tensor(0.0, device=device)
        
        # 2. 反事实一致性损失
        if self.lambda_counterfactual > 0 and labels is not None:
            counterfactual_loss = self.compute_counterfactual_consistency(
                node_embeddings, edge_weights, labels
            )
            losses['counterfactual'] = self.lambda_counterfactual * counterfactual_loss
        else:
            losses['counterfactual'] = torch.tensor(0.0, device=device)
        
        losses['total'] = sum(losses.values())
        
        return losses
    
    def compute_intervention_consistency(
        self,
        node_embeddings: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        干预一致性：do(X_i) 应只影响 X_i 的后代节点
        
        实现：通过邻接矩阵的传播路径判断
        """
        batch_size, N, D = node_embeddings.shape
        device = node_embeddings.device
        
        # 计算可达性矩阵（传播路径）
        adj = (edge_weights > 0.1).float()
        reachability = adj.clone()
        
        # 通过矩阵幂计算传递闭包
        for _ in range(N - 1):
            reachability = torch.clamp(reachability + torch.mm(reachability, adj), 0, 1)
        
        # 对每个节点进行"虚拟干预"
        # 计算干预后的变化是否符合因果结构
        loss = torch.tensor(0.0, device=device)
        
        # 简化：只检查少数节点
        for i in range(min(5, N)):
            # 节点i的后代
            descendants = reachability[i, :] > 0
            non_descendants = ~descendants
            
            # 如果干预节点i，后代应变化，非后代不应变化
            # 这里用嵌入的方差作为"变化"的代理
            desc_var = torch.var(node_embeddings[:, descendants, :], dim=0).mean()
            non_desc_var = torch.var(node_embeddings[:, non_descendants, :], dim=0).mean()
            
            # 惩罚：后代变化小 或 非后代变化大
            loss += F.relu(0.1 - desc_var) + F.relu(non_desc_var - 0.05)
        
        loss /= min(5, N)
        
        return loss
    
    def compute_counterfactual_consistency(
        self,
        node_embeddings: torch.Tensor,
        edge_weights: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        反事实一致性：相同标签的样本应有相似的因果路径
        
        实现：同标签样本的节点嵌入应在关键节点上一致
        """
        batch_size, N, D = node_embeddings.shape
        device = node_embeddings.device
        
        # 按标签分组
        unique_labels = torch.unique(labels)
        
        loss = torch.tensor(0.0, device=device)
        num_pairs = 0
        
        for label in unique_labels:
            mask = (labels == label)
            if mask.sum() < 2:
                continue
            
            # 同标签样本的嵌入
            same_label_embs = node_embeddings[mask]  # [n, N, D]
            
            # 计算两两相似度
            for i in range(len(same_label_embs)):
                for j in range(i + 1, len(same_label_embs)):
                    emb_i = same_label_embs[i]  # [N, D]
                    emb_j = same_label_embs[j]  # [N, D]
                    
                    # 计算每个节点的相似度
                    node_sim = F.cosine_similarity(emb_i, emb_j, dim=-1)  # [N]
                    
                    # 关键节点（高出度）应更相似
                    node_importance = edge_weights.sum(dim=1)  # [N]
                    weighted_sim = (node_importance * node_sim).sum() / (node_importance.sum() + 1e-8)
                    
                    # 惩罚低相似度
                    loss += F.relu(0.7 - weighted_sim)
                    num_pairs += 1
        
        if num_pairs > 0:
            loss /= num_pairs
        
        return loss

