import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


class DAGFusion:
    """
    DAG融合策略
    
    融合方式：
    1. 解释性DAG: 任务级概念/规则节点 (如 blood_pressure → risk_high)
    2. 统计DAG: 数据驱动的标签级节点 (如 yes → no)
    
    融合策略：
    - 线性插值: W_fused = α * W_explain + (1-α) * W_stats
    - 选择性融合: 对解释性DAG中已有边赋予更高权重
    - 层次融合: 解释性DAG在上层(概念),统计DAG在下层(标签)
    """
    
    def __init__(self, fusion_mode='linear', alpha=0.5):
        """
        Args:
            fusion_mode: 融合模式 ('linear', 'selective', 'hierarchical')
            alpha: 解释性DAG的权重 (0.0-1.0)
        """
        self.fusion_mode = fusion_mode
        self.alpha = alpha
    
    def fuse_dags(
        self, 
        explain_dag: Optional[Dict], 
        stats_dag: Dict,
        task_name: str
    ) -> Dict:
        """
        融合解释性DAG和统计DAG
        
        Args:
            explain_dag: 解释性DAG (手动构建,可能为None)
            stats_dag: 统计DAG (自动生成)
            task_name: 任务名称
            
        Returns:
            融合后的DAG字典
        """
        # 如果没有解释性DAG,直接返回统计DAG
        if explain_dag is None or len(explain_dag.get('nodes', [])) == 0:
            return stats_dag
        
        # 如果没有统计DAG,直接返回解释性DAG
        if stats_dag is None or len(stats_dag.get('nodes', [])) == 0:
            return explain_dag
        
        # 根据融合模式选择策略
        if self.fusion_mode == 'linear':
            return self._linear_fusion(explain_dag, stats_dag, task_name)
        elif self.fusion_mode == 'selective':
            return self._selective_fusion(explain_dag, stats_dag, task_name)
        elif self.fusion_mode == 'hierarchical':
            return self._hierarchical_fusion(explain_dag, stats_dag, task_name)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")
    
    def _linear_fusion(
        self, 
        explain_dag: Dict, 
        stats_dag: Dict,
        task_name: str
    ) -> Dict:
        """
        线性融合: W_fused = α * W_explain + (1-α) * W_stats
        
        策略:
        1. 以统计DAG为基础(标签节点)
        2. 如果解释性DAG中存在同名节点的边,则加权融合
        3. 保留解释性DAG中的额外概念节点
        """
        # 提取标签列表
        stats_labels = stats_dag.get('labels', [])
        explain_labels = explain_dag.get('labels', [])
        
        # 构建标签到节点ID的映射
        stats_label_to_id = {str(lbl): i for i, lbl in enumerate(stats_labels)}
        explain_label_nodes = explain_dag.get('label_nodes', {})
        
        # 构建融合后的节点列表 (基于统计DAG的标签节点)
        fused_nodes = list(stats_dag['nodes'])
        node_offset = len(fused_nodes)
        
        # 添加解释性DAG中的概念节点 (非标签节点)
        explain_nodes = explain_dag.get('nodes', [])
        concept_nodes = []
        for node in explain_nodes:
            node_name = node['name']
            # 如果不是标签节点,则为概念节点
            if node_name not in explain_label_nodes or len(explain_label_nodes[node_name]) == 0:
                new_node = node.copy()
                new_node['id'] = node_offset
                concept_nodes.append(new_node)
                node_offset += 1
        
        fused_nodes.extend(concept_nodes)
        
        # 构建边权矩阵
        num_nodes = len(fused_nodes)
        edge_matrix = np.zeros((num_nodes, num_nodes))
        
        # 1. 添加统计DAG的边 (权重: 1-α)
        for edge in stats_dag.get('edges', []):
            src, dst = edge['src'], edge['dst']
            edge_matrix[src, dst] = (1.0 - self.alpha) * edge.get('w', 1.0)
        
        # 2. 添加解释性DAG的边 (权重: α)
        explain_edges = explain_dag.get('edges', [])
        for edge in explain_edges:
            src_id = edge['src']
            dst_id = edge['dst']
            
            # 获取源节点和目标节点的名称
            if src_id >= len(explain_nodes) or dst_id >= len(explain_nodes):
                continue
            
            src_node = explain_nodes[src_id]
            dst_node = explain_nodes[dst_id]
            src_name = src_node['name']
            dst_name = dst_node['name']
            
            # 映射到融合DAG中的节点ID
            # 如果是标签节点,使用stats_label_to_id
            # 如果是概念节点,使用新分配的ID
            fused_src_id = None
            fused_dst_id = None
            
            if src_name in stats_label_to_id:
                fused_src_id = stats_label_to_id[src_name]
            else:
                # 查找概念节点
                for i, cnode in enumerate(concept_nodes):
                    if cnode['name'] == src_name:
                        fused_src_id = len(stats_labels) + i
                        break
            
            if dst_name in stats_label_to_id:
                fused_dst_id = stats_label_to_id[dst_name]
            else:
                # 查找概念节点
                for i, cnode in enumerate(concept_nodes):
                    if cnode['name'] == dst_name:
                        fused_dst_id = len(stats_labels) + i
                        break
            
            if fused_src_id is not None and fused_dst_id is not None:
                edge_matrix[fused_src_id, fused_dst_id] += self.alpha * edge.get('w', 1.0)
        
        # 3. 转换边矩阵为边列表
        fused_edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if edge_matrix[i, j] > 1e-6:
                    fused_edges.append({
                        'src': int(i),
                        'dst': int(j),
                        'w': float(edge_matrix[i, j])
                    })
        
        return {
            'nodes': fused_nodes,
            'edges': fused_edges,
            'labels': stats_labels,
            'label_nodes': {str(lbl): [i] for i, lbl in enumerate(stats_labels)},
            'fusion_info': {
                'mode': 'linear',
                'alpha': self.alpha,
                'num_concept_nodes': len(concept_nodes),
                'num_label_nodes': len(stats_labels)
            }
        }
    
    def _selective_fusion(
        self, 
        explain_dag: Dict, 
        stats_dag: Dict,
        task_name: str
    ) -> Dict:
        """
        选择性融合: 对解释性DAG中已有的边赋予更高权重
        
        策略:
        1. 以统计DAG为基础
        2. 如果边在解释性DAG中存在,使用 α * W_explain + (1-α) * W_stats
        3. 如果边仅在统计DAG中存在,使用 W_stats
        4. 如果边仅在解释性DAG中存在,使用 W_explain
        """
        # 类似linear_fusion,但边权计算策略不同
        # 这里使用线性融合作为基础,可根据需要优化
        return self._linear_fusion(explain_dag, stats_dag, task_name)
    
    def _hierarchical_fusion(
        self, 
        explain_dag: Dict, 
        stats_dag: Dict,
        task_name: str
    ) -> Dict:
        """
        层次融合: 解释性DAG在上层(概念),统计DAG在下层(标签)
        
        策略:
        1. 统计DAG只包含标签节点及其边
        2. 解释性DAG包含概念节点及其到标签节点的连接
        3. 两层通过"概念→标签"边连接
        """
        # 统计DAG的标签节点
        stats_labels = stats_dag.get('labels', [])
        stats_nodes = list(stats_dag['nodes'])
        stats_edges = list(stats_dag['edges'])
        
        # 解释性DAG的概念节点
        explain_nodes = explain_dag.get('nodes', [])
        explain_edges = explain_dag.get('edges', [])
        explain_label_nodes = explain_dag.get('label_nodes', {})
        
        # 构建融合节点: 标签节点 + 概念节点
        fused_nodes = list(stats_nodes)
        node_offset = len(fused_nodes)
        
        concept_nodes = []
        concept_node_map = {}  # 原ID -> 新ID
        
        for node in explain_nodes:
            node_name = node['name']
            # 如果不是标签节点,则为概念节点
            if node_name not in explain_label_nodes or len(explain_label_nodes[node_name]) == 0:
                new_node = node.copy()
                new_id = node_offset
                new_node['id'] = new_id
                concept_nodes.append(new_node)
                concept_node_map[node['id']] = new_id
                node_offset += 1
        
        fused_nodes.extend(concept_nodes)
        
        # 构建融合边
        fused_edges = []
        
        # 1. 添加标签层的边 (统计DAG)
        for edge in stats_edges:
            fused_edges.append(edge.copy())
        
        # 2. 添加概念层的边 (解释性DAG)
        stats_label_to_id = {str(lbl): i for i, lbl in enumerate(stats_labels)}
        
        for edge in explain_edges:
            src_id = edge['src']
            dst_id = edge['dst']
            
            if src_id >= len(explain_nodes) or dst_id >= len(explain_nodes):
                continue
            
            src_node = explain_nodes[src_id]
            dst_node = explain_nodes[dst_id]
            src_name = src_node['name']
            dst_name = dst_node['name']
            
            # 映射节点ID
            fused_src_id = concept_node_map.get(src_id)
            if fused_src_id is None and src_name in stats_label_to_id:
                fused_src_id = stats_label_to_id[src_name]
            
            fused_dst_id = concept_node_map.get(dst_id)
            if fused_dst_id is None and dst_name in stats_label_to_id:
                fused_dst_id = stats_label_to_id[dst_name]
            
            if fused_src_id is not None and fused_dst_id is not None:
                fused_edges.append({
                    'src': int(fused_src_id),
                    'dst': int(fused_dst_id),
                    'w': float(edge.get('w', 1.0))
                })
        
        return {
            'nodes': fused_nodes,
            'edges': fused_edges,
            'labels': stats_labels,
            'label_nodes': {str(lbl): [i] for i, lbl in enumerate(stats_labels)},
            'fusion_info': {
                'mode': 'hierarchical',
                'num_concept_nodes': len(concept_nodes),
                'num_label_nodes': len(stats_labels)
            }
        }


def load_explain_dag(task_name: str, manual_dag_file: str) -> Optional[Dict]:
    """
    加载解释性DAG
    
    Args:
        task_name: 任务名称
        manual_dag_file: DAG文件路径
        
    Returns:
        解释性DAG字典,如果不存在则返回None
    """
    import json
    import os
    
    if not os.path.exists(manual_dag_file):
        return None
    
    try:
        with open(manual_dag_file, 'r', encoding='utf-8') as f:
            all_dags = json.load(f)
        
        if task_name in all_dags:
            return all_dags[task_name]
        else:
            return None
    except Exception as e:
        print(f"⚠️  加载解释性DAG失败: {e}")
        return None

