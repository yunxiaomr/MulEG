"""
改进版CausalGNNClues包装器
集成所有改进模块：
1. 混合编码器 (HybridEncoder) - 文本+数值+符号
2. 序损失 (OrdinalRegressionLoss)
3. 任务自适应Dropout
4. 数值感知对比学习
5. 任务嵌入与Adapter
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import json
import os

# 导入原始wrapper
from .clues_wrapper import CausalGNNClues as BaseCausalGNNClues

# 导入新模块
from .hybrid_encoder import HybridEncoder, identify_numeric_features
from .ordinal_loss import (
    OrdinalRegressionLoss, 
    detect_ordinal_labels,
    create_ordinal_label_map,
    get_ordinal_task_info
)
from .task_adaptive_dropout import (
    TaskAdaptiveDropout,
    load_task_types,
    initialize_task_dropouts
)
from .numerical_contrastive_loss import (
    NumericalContrastiveLoss,
    TaskContrastiveLoss
)
from .symbolic_encoder import (
    SymbolicGraphEncoder,
    HybridEncoderWithSymbolic,
    encode_tic_tac_toe_as_graph,
    add_rule_prompt,
    is_symbolic_task
)
from .task_embedding import (
    TaskEmbeddingWithAdapter
)


class ImprovedCausalGNNClues(BaseCausalGNNClues):
    """
    改进版CausalGNNClues，集成所有新功能
    """
    
    def __init__(self, encoder, tokenizer, dag_per_task, hidden_dim=256,
                 num_heads=4, num_layers=2, kappa=1.0, label_details_path=None,
                 use_token_sim=False, use_bert_token_sim=False, 
                 use_enhancement=False, enhancement_config=None,
                 # 新增参数
                 use_hybrid_encoder=True,
                 use_cross_attention=True,
                 use_ordinal_loss=True,
                 use_adaptive_dropout=True,
                 use_numerical_contrastive=True,
                 use_task_contrastive=False,
                 use_symbolic_encoder=False,
                 use_task_embedding=True,
                 use_task_adapter=True,
                 task_type_json=None,
                 real_task_details_json=None,
                 adapter_dim=64,
                 ordinal_loss_alpha=1.0,
                 ordinal_loss_beta=0.5,
                 numerical_contrastive_temp=0.07,
                 numerical_contrastive_weight=0.5,
                 task_contrastive_temp=0.07):
        """
        Args:
            ... (继承基类参数) ...
            use_hybrid_encoder: 是否使用混合编码器
            use_cross_attention: 混合编码器中是否使用跨模态注意力
            use_ordinal_loss: 是否使用序损失
            use_adaptive_dropout: 是否使用任务自适应dropout
            use_numerical_contrastive: 是否使用数值感知对比学习
            use_task_contrastive: 是否使用跨任务对比学习
            use_symbolic_encoder: 是否使用符号编码器
            use_task_embedding: 是否使用任务嵌入
            use_task_adapter: 是否使用任务adapter
            task_type_json: 任务类型JSON文件路径
            real_task_details_json: 任务详情JSON文件路径
            adapter_dim: Adapter降维维度
            ordinal_loss_alpha: 序损失的rank权重
            ordinal_loss_beta: 序损失的distance权重
            numerical_contrastive_temp: 数值对比学习温度
            numerical_contrastive_weight: 数值相似性权重
            task_contrastive_temp: 跨任务对比学习温度
        """
        # 初始化基类
        super().__init__(encoder, tokenizer, dag_per_task, hidden_dim,
                        num_heads, num_layers, kappa, label_details_path,
                        use_token_sim, use_bert_token_sim, 
                        use_enhancement, enhancement_config)
        
        # 保存配置
        self.use_hybrid_encoder = use_hybrid_encoder
        self.use_ordinal_loss = use_ordinal_loss
        self.use_adaptive_dropout = use_adaptive_dropout
        self.use_numerical_contrastive = use_numerical_contrastive
        self.use_task_contrastive = use_task_contrastive
        self.use_symbolic_encoder = use_symbolic_encoder
        self.use_task_embedding = use_task_embedding
        self.use_task_adapter = use_task_adapter
        
        # 1. 混合编码器
        if self.use_hybrid_encoder:
            if self.use_symbolic_encoder:
                self.hybrid_encoder = HybridEncoderWithSymbolic(
                    bert_encoder=encoder,
                    embed_dim=self.embed_dim
                )
                print(f"[ImprovedCausalGNN] 已启用混合编码器 (文本+数值+符号)")
            else:
                self.hybrid_encoder = HybridEncoder(
                    bert_encoder=encoder,
                    embed_dim=self.embed_dim,
                    use_cross_attention=use_cross_attention
                )
                print(f"[ImprovedCausalGNN] 已启用混合编码器 (文本+数值)")
        
        # 2. 任务类型映射和序损失
        self.task_type_map = {}
        self.ordinal_tasks = {}  # {task_name: {'label_map': ..., 'loss_fn': ...}}
        
        if task_type_json and os.path.exists(task_type_json):
            self.task_type_map = load_task_types(task_type_json)
            print(f"[ImprovedCausalGNN] 已加载任务类型映射: {len(self.task_type_map)}个任务")
        
        if use_ordinal_loss and real_task_details_json and os.path.exists(real_task_details_json):
            self._initialize_ordinal_losses(real_task_details_json, 
                                          ordinal_loss_alpha, 
                                          ordinal_loss_beta)
        
        # 3. 任务自适应Dropout
        if self.use_adaptive_dropout:
            self.dropout_controller = TaskAdaptiveDropout(default_dropout=0.15)
            if task_type_json:
                initialize_task_dropouts(self.dropout_controller, task_type_json)
            print(f"[ImprovedCausalGNN] 已启用任务自适应Dropout")
        
        # 4. 数值感知对比学习
        if self.use_numerical_contrastive:
            self.numerical_contrastive_loss = NumericalContrastiveLoss(
                temperature=numerical_contrastive_temp,
                numerical_weight=numerical_contrastive_weight
            )
            print(f"[ImprovedCausalGNN] 已启用数值感知对比学习")
        
        # 5. 跨任务对比学习
        if self.use_task_contrastive:
            self.task_contrastive_loss = TaskContrastiveLoss(
                temperature=task_contrastive_temp
            )
            print(f"[ImprovedCausalGNN] 已启用跨任务对比学习")
        
        # 6. 任务嵌入与Adapter
        if self.use_task_embedding or self.use_task_adapter:
            self.task_embedding_module = TaskEmbeddingWithAdapter(
                num_tasks=36,
                embed_dim=self.embed_dim,
                adapter_dim=adapter_dim,
                use_adapter=self.use_task_adapter
            )
            print(f"[ImprovedCausalGNN] 已启用任务嵌入{'与Adapter' if self.use_task_adapter else ''}")
    
    def _initialize_ordinal_losses(self, real_task_details_json: str,
                                   alpha: float, beta: float):
        """初始化序损失函数"""
        ordinal_info = get_ordinal_task_info(real_task_details_json)
        
        for task_name, info in ordinal_info.items():
            num_classes = info['num_classes']
            ordered_labels = info['ordered_labels']
            
            # 创建损失函数
            loss_fn = OrdinalRegressionLoss(
                num_classes=num_classes,
                alpha=alpha,
                beta=beta
            )
            
            # 创建标签映射
            label_to_idx = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
            
            self.ordinal_tasks[task_name] = {
                'num_classes': num_classes,
                'ordered_labels': ordered_labels,
                'label_to_idx': label_to_idx,
                'loss_fn': loss_fn
            }
        
        print(f"[ImprovedCausalGNN] 已为 {len(self.ordinal_tasks)} 个序数任务初始化序损失")
        for task_name, info in list(self.ordinal_tasks.items())[:5]:
            print(f"  {task_name}: {info['ordered_labels']}")
    
    def register_task(self, task_name: str):
        """注册任务（用于任务嵌入和adapter）"""
        if self.use_task_embedding or self.use_task_adapter:
            self.task_embedding_module.register_task(task_name)
    
    def forward(self, input_ids, attention_mask, token_type_ids,
                task_name, target_text_matrix=None, entail_indicators=None,
                guideline_inputs=None, labels=None, 
                numeric_features=None, feature_mask=None,
                return_loss=False):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            task_name: str 任务名称
            target_text_matrix: 目标文本矩阵（兼容原始接口）
            entail_indicators: 蕴含指示器（兼容原始接口）
            guideline_inputs: 指南输入（兼容原始接口）
            labels: (batch_size,) 真实标签（可选）
            numeric_features: (batch_size, num_features) 数值特征（可选）
            feature_mask: (num_features,) bool mask（可选）
            return_loss: 是否返回损失
        
        Returns:
            与BaseCausalGNNClues兼容的格式: (logits, aux_dict)
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        # 注册任务
        self.register_task(task_name)
        
        # 智能编码器选择：只对纯数值任务使用混合编码器
        task_type = self.task_type_map.get(task_name, 'semantic')
        
        # 判断是否使用混合编码器
        use_hybrid = False
        if self.use_hybrid_encoder and task_type == 'numerical':
            # 只对纯数值任务使用混合编码器
            # 检查是否有足够的数值特征
            if numeric_features is not None and feature_mask is not None:
                num_numeric_features = feature_mask.sum().item() if isinstance(feature_mask, torch.Tensor) else sum(feature_mask)
                if num_numeric_features >= 5:  # 至少5个数值特征才使用混合编码器
                    use_hybrid = True
        
        if use_hybrid:
            text_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            
            sample_embed = self.hybrid_encoder(
                text_input=text_input,
                numeric_features=numeric_features,
                feature_mask=feature_mask
            )
        else:
            # 其他任务只用BERT
            sample_embed = self.encode_text(input_ids, attention_mask, token_type_ids)
        
        # 应用任务嵌入和Adapter
        if self.use_task_embedding or self.use_task_adapter:
            sample_embed = self.task_embedding_module(sample_embed, task_name)
        
        # 应用任务自适应Dropout
        if self.use_adaptive_dropout and self.training:
            task_type = self.task_type_map.get(task_name, 'semantic')
            self.dropout_controller.set_task_dropout(task_name, task_type)
            sample_embed = self.dropout_controller(sample_embed, task_name)
        
        # 获取DAG和标签原型
        dag = self.dag_per_task.get(task_name, None)
        if dag is None:
            raise ValueError(f"Task {task_name} not found in dag_per_task")
        
        # 确保DAG节点有text_embed（手工DAG载入时为None，需要即刻编码）
        if any(n.text_embed is None for n in dag.nodes):
            texts = [n.text for n in dag.nodes]
            with torch.no_grad():  # DAG结构embedding不需要梯度
                enc_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
                enc_outputs = self.encoder(**enc_inputs)
                node_emb = enc_outputs.pooler_output
            for i, n in enumerate(dag.nodes):
                n.text_embed = node_emb[i].detach()  # detach确保不会多次backward
        
        # 标签原型相似度
        label_details = self.label_details.get(task_name, {}) if self.label_details else {}
        label_names = dag.labels  # CausalDAG使用.labels属性
        
        # 关键修复：标签原型编码需要参与梯度，但不能每次都重新编码同一个tensor
        # 方案：每次编码新的tensor，而不是复用之前的
        label_embeds = []
        for lbl in label_names:
            if label_details and lbl in label_details:
                proto_texts = label_details[lbl]
                if not isinstance(proto_texts, list):
                    proto_texts = [proto_texts]
            else:
                proto_texts = [lbl]
            
            # 每次都重新编码，生成新的tensor（不会导致重复backward）
            proto_inputs = self.tokenizer(
                proto_texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            proto_inputs = {k: v.to(device) for k, v in proto_inputs.items()}
            
            # 编码标签原型（每个batch重新编码，参与梯度计算）
            proto_outputs = self.encoder(**proto_inputs)
            proto_embed = proto_outputs.pooler_output.mean(dim=0)
            
            label_embeds.append(proto_embed)
        
        label_embeds = torch.stack(label_embeds)  # (num_labels, embed_dim)
        
        # 使用基类的GNN逻辑
        # 获取DAG节点embeddings
        node_emb = torch.stack([n.text_embed for n in dag.nodes], dim=0)  # [N, D]
        node_emb_norm = F.normalize(node_emb, dim=-1)
        
        # 计算sample与所有节点的相似度
        sample_embed_norm = F.normalize(sample_embed, dim=-1)  # [B, D]
        s_v = torch.matmul(sample_embed_norm, node_emb_norm.t())  # [B, N]
        
        # GNN forward（逐样本处理）
        node_logits_list = []
        attn_entropies = []
        
        for b in range(batch_size):
            s_v_b = s_v[b]  # [N] - 所有节点都有相似度值
            node_logits_b, attn_e = self.gnn(dag, s_v_b)
            node_logits_list.append(node_logits_b)
            attn_entropies.extend(attn_e)
        
        node_logits_batch = torch.stack(node_logits_list, dim=0)  # (B, N)
        
        # 零样本Readout: 概念增强 + 标签原型匹配
        # ===============================================================
        #   - 手动DAG只包含概念/特征节点，不包含标签节点（零样本原则）
        #   - GNN在概念DAG上传播，学习概念之间的因果关系
        #   - 使用GNN输出的概念表示增强样本embedding
        #   - 增强后的样本与标签原型匹配，选择最相似的标签
        # ===============================================================
        
        # 使用GNN输出增强样本表示
        # node_logits_batch: (B, N) - 每个样本在每个概念节点上的激活强度
        
        # 方案：Top-K注意力加权池化
        # 选择激活最高的K个概念节点，用它们的表示增强样本
        # 优先从任务级配置读取TopK
        task_topk_map = getattr(self, 'zs_task_topk_map', None)
        if isinstance(task_topk_map, dict) and task_name in task_topk_map:
            # 使用任务级配置的TopK（绝对值）
            k = int(task_topk_map[task_name])
            topk_ratio = k / len(dag.nodes) if len(dag.nodes) > 0 else 0.5
        else:
            # 使用默认比例
            topk_ratio = 0.5  # 默认使用top 50%的节点
            k = max(1, int(len(dag.nodes) * topk_ratio))
        
        k = min(k, len(dag.nodes))
        k = max(1, k)  # 至少选择1个节点
        
        # 对每个样本，找到top-k激活的概念节点
        topk_values, topk_indices = torch.topk(node_logits_batch, k, dim=1)  # (B, k)
        
        # 使用softmax归一化top-k激活作为注意力权重
        attn_weights = F.softmax(topk_values, dim=1)  # (B, k)
        
        # 获取top-k概念节点的embedding
        node_emb = torch.stack([n.text_embed for n in dag.nodes], dim=0)  # (N, D)
        topk_node_emb = torch.stack([
            node_emb[topk_indices[b]] for b in range(batch_size)
        ], dim=0)  # (B, k, D)
        
        # 注意力加权池化得到GNN增强的表示
        gnn_enhanced_emb = torch.sum(attn_weights.unsqueeze(-1) * topk_node_emb, dim=1)  # (B, D)
        
        # 融合原始样本表示和GNN增强表示
        # fusion_weight: GNN增强表示的权重（0=纯BERT, 1=纯GNN）
        fusion_weight = 0.5  # 默认值
        if hasattr(self, 'zs_task_mix_alpha') and task_name in self.zs_task_mix_alpha:
            fusion_weight = self.zs_task_mix_alpha[task_name]
        
        fused_sample_emb = (1 - fusion_weight) * sample_embed + fusion_weight * gnn_enhanced_emb
        fused_sample_emb_norm = F.normalize(fused_sample_emb, dim=-1)
        
        # 零样本标签匹配：融合后的样本与标签原型计算相似度
        label_embeds_norm = F.normalize(label_embeds, dim=-1)
        logits = torch.matmul(fused_sample_emb_norm, label_embeds_norm.t())  # (B, num_labels)
        
        # 调试输出（偶尔打印）
        if self.training and torch.rand(1).item() < 0.001:  # 0.1%概率
            print(f"\n[Zero-Shot Readout] Task: {task_name}")
            print(f"  概念节点数: {len(dag.nodes)}")
            print(f"  Top-K: {k} (ratio={topk_ratio})")
            print(f"  融合权重: BERT={1-fusion_weight:.3f}, GNN={fusion_weight:.3f}")
            print(f"  Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
        
        # 构建辅助字典（兼容原始接口）
        aux = {
            'attn_entropies': attn_entropies,  # 注意力熵
            'enhancement_losses': {},  # 增强损失（占位）
        }
        
        # 计算改进模块的损失（如果在训练阶段）
        if labels is not None:
            # 主损失：序损失或交叉熵（将在外部计算）
            # 这里我们将改进模块的额外损失添加到aux中
            improvement_losses = {}
            
            # 数值感知对比学习损失
            if self.use_numerical_contrastive and numeric_features is not None:
                try:
                    num_contrast_loss = self.numerical_contrastive_loss(
                        sample_embed, numeric_features, labels
                    )
                    improvement_losses['numerical_contrastive'] = num_contrast_loss
                except Exception as e:
                    print(f"Warning: numerical contrastive loss failed: {e}")
                    improvement_losses['numerical_contrastive'] = torch.tensor(0.0, device=device)
            
            # 存储改进损失到aux中
            aux['improvement_losses'] = improvement_losses
            
            # 存储序损失信息（如果适用）
            if self.use_ordinal_loss and task_name in self.ordinal_tasks:
                aux['use_ordinal_loss'] = True
                aux['ordinal_task_info'] = self.ordinal_tasks[task_name]
                aux['label_names'] = label_names
            else:
                aux['use_ordinal_loss'] = False
        
        # 返回与BaseCausalGNNClues兼容的格式
        return logits, aux
    
    def compute_total_loss(self, loss_dict: Dict[str, torch.Tensor],
                          lambda_numerical_contrastive: float = 0.1,
                          lambda_task_contrastive: float = 0.05) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            loss_dict: 损失字典
            lambda_numerical_contrastive: 数值对比学习权重
            lambda_task_contrastive: 跨任务对比学习权重
        
        Returns:
            total_loss: 总损失
        """
        total_loss = loss_dict['main_loss']
        
        if 'numerical_contrastive_loss' in loss_dict:
            total_loss = total_loss + lambda_numerical_contrastive * loss_dict['numerical_contrastive_loss']
        
        if 'task_contrastive_loss' in loss_dict:
            total_loss = total_loss + lambda_task_contrastive * loss_dict['task_contrastive_loss']
        
        return total_loss

