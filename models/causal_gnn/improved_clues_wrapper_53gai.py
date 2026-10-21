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
                 use_focal_loss=True,
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
                 task_contrastive_temp=0.07,
                 # 补充可能缺失的Focal Loss参数默认值
                 focal_loss_alpha=0.25,
                 focal_loss_gamma=2.0,
                 focal_loss_num_classes_threshold=4):
        """
        Args:
            ... (继承基类参数) ...
        """
        # 初始化基类
        super().__init__(encoder, tokenizer, dag_per_task, hidden_dim,
                        num_heads, num_layers, kappa, label_details_path,
                        use_token_sim, use_bert_token_sim,
                        use_enhancement, enhancement_config)

        # 解释层级消融控制
        self.explanation_ablation = 'full'  # 'full' | 'sample_only' | 'attribute_only' | 'label_only' | 'sample_attribute' | 'sample_label' | 'attribute_label'

        # 保存配置
        self.use_hybrid_encoder = use_hybrid_encoder
        self.use_ordinal_loss = use_ordinal_loss
        self.use_focal_loss = use_focal_loss
        self.use_adaptive_dropout = use_adaptive_dropout
        self.use_numerical_contrastive = use_numerical_contrastive
        self.use_task_contrastive = use_task_contrastive
        self.use_symbolic_encoder = use_symbolic_encoder
        self.use_task_embedding = use_task_embedding
        self.use_task_adapter = use_task_adapter
        
        # 1. 混合编码器初始化
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
        
        # 2. 任务类型映射加载 (含 Point 1 & V55 JSON 兼容逻辑)
        self.task_type_map = {}
        self.ordinal_tasks = {}
        
        if task_type_json and os.path.exists(task_type_json):
            try:
                with open(task_type_json, 'r') as f:
                    raw_data = json.load(f)
                # 兼容 {"type": ["task1"...]} 格式 (V55格式) 或扁平格式
                if raw_data:
                    first_val = next(iter(raw_data.values()))
                    if isinstance(first_val, list):
                        for t_type, t_list in raw_data.items():
                            for t_name in t_list:
                                self.task_type_map[t_name] = t_type
                    else:
                        self.task_type_map = raw_data     
                print(f"[ImprovedCausalGNN] 已加载并解析任务类型映射: 覆盖 {len(self.task_type_map)} 个任务")
            except Exception as e:
                print(f"[ImprovedCausalGNN] ⚠️ 加载任务类型映射失败: {e}")
        
        # 3. 初始化序损失
        if use_ordinal_loss and real_task_details_json and os.path.exists(real_task_details_json):
            self._initialize_ordinal_losses(
                real_task_details_json, 
                ordinal_loss_alpha, 
                ordinal_loss_beta
            )
        
        # 4. 任务自适应Dropout
        if self.use_adaptive_dropout:
            self.dropout_controller = TaskAdaptiveDropout(default_dropout=0.15)
            if task_type_json:
                initialize_task_dropouts(self.dropout_controller, task_type_json)
            print(f"[ImprovedCausalGNN] 已启用任务自适应Dropout")
        
        # 5. 数值感知对比学习
        if self.use_numerical_contrastive:
            self.numerical_contrastive_loss = NumericalContrastiveLoss(
                temperature=numerical_contrastive_temp,
                numerical_weight=numerical_contrastive_weight
            )
            print(f"[ImprovedCausalGNN] 已启用数值感知对比学习")
        
        # 6. 跨任务对比学习
        if self.use_task_contrastive:
            self.task_contrastive_loss = TaskContrastiveLoss(
                temperature=task_contrastive_temp
            )
            print(f"[ImprovedCausalGNN] 已启用跨任务对比学习")
        
        # 7. 任务嵌入与Adapter
        if self.use_task_embedding or self.use_task_adapter:
            self.task_embedding_module = TaskEmbeddingWithAdapter(
                num_tasks=41,
                embed_dim=self.embed_dim,
                adapter_dim=adapter_dim,
                use_adapter=self.use_task_adapter
            )
            print(f"[ImprovedCausalGNN] 已启用任务嵌入{'与Adapter' if self.use_task_adapter else ''}")

        # 缓存无边的 DAG (用于 attribute-level ablation)
        self._no_edge_dags = {}

    def _get_ablation_dag(self, task_name):
        """获取用于消融的 DAG (可选移除边)."""
        dag = self.dag_per_task.get(task_name)
        if dag is None:
            return None
        if self._use_attribute_level:
            return dag
        # 返回无边版本 (缓存)
        if task_name not in self._no_edge_dags:
            from models.causal_graph.schemas import CausalDAG
            no_edge_dag = CausalDAG(
                nodes=list(dag.nodes),
                edges=[],
                labels=list(dag.labels) if dag.labels else [],
                label_nodes=dict(dag.label_nodes) if dag.label_nodes else {}
            )
            self._no_edge_dags[task_name] = no_edge_dag
        return self._no_edge_dags[task_name]

    def _initialize_ordinal_losses(self, real_task_details_json: str, alpha: float, beta: float):
        """初始化序损失函数"""
        ordinal_info = get_ordinal_task_info(real_task_details_json)
        # 找出所有在 task_type_map 中被声明为 'ordinal' 的任务
        declared_ordinal_tasks = [t_name for t_name, t_type in self.task_type_map.items() if t_type == 'ordinal']
        
        for task_name in declared_ordinal_tasks:
            if task_name not in ordinal_info:
                continue # Skip safety check
            
            info = ordinal_info[task_name]
            num_classes = info['num_classes']
            ordered_labels = info['ordered_labels']
            loss_fn = OrdinalRegressionLoss(num_classes=num_classes, alpha=alpha, beta=beta)
            label_to_idx = {lbl: idx for idx, lbl in enumerate(ordered_labels)}
            
            self.ordinal_tasks[task_name] = {
                'num_classes': num_classes,
                'ordered_labels': ordered_labels,
                'label_to_idx': label_to_idx,
                'loss_fn': loss_fn
            }
        print(f"[ImprovedCausalGNN] 已成功为 {len(self.ordinal_tasks)} 个序数任务初始化序损失")

    def set_explanation_ablation(self, mode: str):
        """设置解释层级消融模式.

        Args:
            mode: 'full' | 'sample_only' | 'attribute_only' | 'label_only' |
                  'sample_attribute' | 'sample_label' | 'attribute_label'
        """
        valid_modes = ['full', 'sample_only', 'attribute_only', 'label_only',
                       'sample_attribute', 'sample_label', 'attribute_label']
        if mode not in valid_modes:
            raise ValueError(f"Invalid explanation_ablation mode: {mode}. Choose from {valid_modes}")
        self.explanation_ablation = mode
        # 解析哪些层级启用
        self._use_sample_level = mode in ['full', 'sample_only', 'sample_attribute', 'sample_label']
        self._use_attribute_level = mode in ['full', 'attribute_only', 'sample_attribute', 'attribute_label']
        self._use_label_level = mode in ['full', 'label_only', 'sample_label', 'attribute_label']
        print(f"[ImprovedCausalGNN] 解释层级消融: {mode} "
              f"(sample={self._use_sample_level}, attr={self._use_attribute_level}, label={self._use_label_level})")

    def set_view_mode(self, mode: str):
        """设置数值特征编码的视图模式 (用于消融实验)."""
        if hasattr(self, 'hybrid_encoder') and self.use_hybrid_encoder:
            self.hybrid_encoder.set_view_mode(mode)
            print(f"[ImprovedCausalGNN] 视图模式: {mode}")

    def register_task(self, task_name: str):
        if self.use_task_embedding or self.use_task_adapter:
            self.task_embedding_module.register_task(task_name)
    
    def forward(self, input_ids, attention_mask, token_type_ids,
                task_name, target_text_matrix=None, entail_indicators=None,
                guideline_inputs=None, labels=None, 
                numeric_features=None, feature_mask=None,
                return_loss=False):
        
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Task Name Unboxing
        if isinstance(task_name, (list, tuple)):
            task_name = task_name[0]
        
        # NaN Safety
        if numeric_features is not None:
            numeric_features = torch.nan_to_num(numeric_features, nan=0.0)

        self.register_task(task_name)
        
        # [修改点: Point 2, 8, 9] 改进的 Hybrid 触发逻辑
        task_type = self.task_type_map.get(task_name, 'semantic')
        use_hybrid = False
        
        # Point 2: 允许 mixed, numerical, symbolic 使用，移除 feat>=5 限制
        if self.use_hybrid_encoder and task_type in ['numerical', 'mixed', 'symbolic']:
            # 只要有特征数据就启用，不设高门槛
            if numeric_features is not None:
                has_features = False
                # 检查 tensor 维度和内容
                if numeric_features.dim() > 1 and numeric_features.size(1) > 0:
                    has_features = True
                
                if has_features:
                    use_hybrid = True
        
        # Point 8: 激活 Tic-Tac-Toe
        if self.use_hybrid_encoder and 'tic-tac-toe' in task_name:
            use_hybrid = True
            
        # Point 9: 激活 Driving Championship
        if self.use_hybrid_encoder and 'driving-championship' in task_name:
            use_hybrid = True

        # 如果开启了符号编码器且是符号任务，也强制开启
        if self.use_hybrid_encoder and self.use_symbolic_encoder and task_type == 'symbolic':
            use_hybrid = True

        if use_hybrid:
            text_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            
            # [修改点: Point 4] Feature Mask 维度防御性检查
            feature_mask_arg = feature_mask
            if feature_mask is not None and isinstance(feature_mask, torch.Tensor) and feature_mask.dim() > 1:
                # 如果是 (B, NumFeatures) 且 batch 内 mask 一致，取第一行
                feature_mask_arg = feature_mask[0]
            
            sample_embed = self.hybrid_encoder(
                text_input=text_input,
                numeric_features=numeric_features,
                feature_mask=feature_mask_arg 
            )
        else:
            sample_embed = self.encode_text(input_ids, attention_mask, token_type_ids)
        
        # 2. 应用任务嵌入和Adapter
        if self.use_task_embedding or self.use_task_adapter:
            sample_embed = self.task_embedding_module(sample_embed, task_name)
        
        # 3. 应用任务自适应Dropout
        if self.use_adaptive_dropout and self.training:
            task_type = self.task_type_map.get(task_name, 'semantic')
            self.dropout_controller.set_task_dropout(task_name, task_type)
            sample_embed = self.dropout_controller(sample_embed, task_name)
        
        # 4. 获取DAG (支持消融: attribute-level off 时移除边)
        dag = self.dag_per_task.get(task_name, None)
        if dag is None:
            raise ValueError(f"Task {task_name} not found in dag_per_task")
        ablation_dag = self._get_ablation_dag(task_name) if hasattr(self, '_use_attribute_level') else dag

        # 5. DAG Text Embed (一次性编码)
        if any(n.text_embed is None for n in dag.nodes):
            texts = [n.text for n in dag.nodes]
            with torch.no_grad():
                enc_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
                enc_outputs = self.encoder(**enc_inputs)
                node_emb = enc_outputs.pooler_output
            for i, n in enumerate(dag.nodes):
                n.text_embed = node_emb[i].detach()
        
        node_emb = torch.stack([n.text_embed for n in dag.nodes], dim=0)
        
        # 增强模块 (保持原样)
        enhancement_losses = {}
        if self.use_enhancement:
            sample_embed_enhanced = self.enhancement.apply_whitening(sample_embed, update_stats=self.training)
            node_emb_enhanced = self.enhancement.apply_whitening(node_emb, update_stats=self.training)
            if self.training:
                sample_embed_enhanced = self.enhancement.apply_noise(sample_embed_enhanced, training=True)
                node_emb_enhanced = self.enhancement.apply_noise(node_emb_enhanced, training=True)
                iso_loss_sample = self.enhancement.compute_isotropy_loss(sample_embed_enhanced)
                iso_loss_node = self.enhancement.compute_isotropy_loss(node_emb_enhanced)
                enhancement_losses['isotropy'] = iso_loss_sample + iso_loss_node
            sample_embed_for_sim = sample_embed_enhanced
            node_emb_for_sim = node_emb_enhanced
        else:
            sample_embed_for_sim = sample_embed
            node_emb_for_sim = node_emb

        # 6. GNN
        node_emb_norm = F.normalize(node_emb_for_sim, dim=-1)
        sample_embed_norm = F.normalize(sample_embed_for_sim, dim=-1)
        
        if self.use_enhancement and self.training:
             sim_ent_loss = self.enhancement.compute_similarity_entropy_loss(node_emb_for_sim, sample_embed_for_sim)
             enhancement_losses['similarity_entropy'] = sim_ent_loss

        # s_v: sample-node similarity (sample-level explanation)
        if getattr(self, '_use_sample_level', True):
            s_v = torch.matmul(sample_embed_norm, node_emb_norm.t())
        else:
            s_v = torch.ones(batch_size, node_emb_norm.size(0), device=device) / node_emb_norm.size(0)
        
        node_logits_list = []
        attn_entropies = []

        attr_enabled = getattr(self, '_use_attribute_level', True)
        for b in range(batch_size):
            s_v_b = s_v[b]
            if attr_enabled:
                node_logits_b, attn_e = self.gnn(ablation_dag, s_v_b)
                attn_entropies.extend(attn_e)
            else:
                # 属性级消融: 跳过 GNN 消息传递，直接用 s_v 作为每节点分数
                # 应用简单的线性变换保持维度一致
                node_logits_b = self.gnn.readout(
                    self.gnn.input_proj(
                        torch.cat([node_emb, (self.gnn.kappa * s_v_b)[:, None]], dim=-1)
                    )
                ).squeeze(-1)
            node_logits_list.append(node_logits_b)
        node_logits_batch = torch.stack(node_logits_list, dim=0)
        
        # 8. Zero-Shot Readout (V53 的 GNN 增强逻辑)
        # 这里保留 V53/V55 的核心逻辑：TopK 节点增强
        k = max(1, int(len(dag.nodes) * 0.5)) 
        
        topk_values, topk_indices = torch.topk(node_logits_batch, k, dim=1)
        attn_weights = F.softmax(topk_values, dim=1)
        topk_node_emb = torch.stack([node_emb[topk_indices[b]] for b in range(batch_size)], dim=0)
        gnn_enhanced_emb = torch.sum(attn_weights.unsqueeze(-1) * topk_node_emb, dim=1)
        
        fusion_weight = 0.5
        if hasattr(self, 'zs_task_mix_alpha_map') and task_name in self.zs_task_mix_alpha_map:
            fusion_weight = self.zs_task_mix_alpha_map[task_name]
        
        fused_sample_emb = (1 - fusion_weight) * sample_embed + fusion_weight * gnn_enhanced_emb
        fused_sample_emb_norm = F.normalize(fused_sample_emb, dim=-1)
        
        # Label 显式重编码 (支持消融: label-level off 时使用简单标签名)
        use_label_proto = getattr(self, '_use_label_level', True)
        label_details = self.label_details.get(task_name, {}) if (self.label_details and use_label_proto) else {}
        label_names = dag.labels
        label_embeds = []
        
        for lbl in label_names:
            if label_details and lbl in label_details:
                proto_texts = label_details[lbl] if isinstance(label_details[lbl], list) else [label_details[lbl]]
            else:
                proto_texts = [lbl]
            
            # 每次前向传播都重新编码，确保计算图完整
            proto_inputs = self.tokenizer(proto_texts, padding=True, truncation=True, return_tensors='pt')
            proto_inputs = {k: v.to(device) for k, v in proto_inputs.items()}
            proto_outputs = self.encoder(**proto_inputs)
            label_embeds.append(proto_outputs.pooler_output.mean(dim=0))
        
        label_embeds = torch.stack(label_embeds)
        label_embeds_norm = F.normalize(label_embeds, dim=-1)
        logits = torch.matmul(fused_sample_emb_norm, label_embeds_norm.t())
        
        # Temperature Scaling
        tau = 0.5
        if hasattr(self, 'zs_task_tau_map') and isinstance(self.zs_task_tau_map, dict):
            tau = float(self.zs_task_tau_map.get(task_name, 0.5))
        logits = logits / tau
        # 数值稳定性安全网
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)

        aux = {
            'attn_entropies': attn_entropies,
            'enhancement_losses': enhancement_losses,
        }
        
        if labels is not None:
            improvement_losses = {}
            if self.use_numerical_contrastive and numeric_features is not None:
                try:
                    num_contrast_loss = self.numerical_contrastive_loss(sample_embed, numeric_features, labels)
                    improvement_losses['numerical_contrastive'] = num_contrast_loss
                except Exception:
                    improvement_losses['numerical_contrastive'] = torch.tensor(0.0, device=device)
            aux['improvement_losses'] = improvement_losses
            
            if self.use_ordinal_loss and task_name in self.ordinal_tasks:
                aux['use_ordinal_loss'] = True
                aux['ordinal_task_info'] = self.ordinal_tasks[task_name]
                aux['label_names'] = label_names
            else:
                aux['use_ordinal_loss'] = False
        
        return logits, aux

    def compute_total_loss(self, loss_dict: Dict[str, torch.Tensor],
                          lambda_numerical_contrastive: float = 0.1,
                          lambda_task_contrastive: float = 0.05) -> torch.Tensor:
        total_loss = loss_dict['main_loss']
        if 'numerical_contrastive_loss' in loss_dict:
            total_loss = total_loss + lambda_numerical_contrastive * loss_dict['numerical_contrastive_loss']
        if 'task_contrastive_loss' in loss_dict:
            total_loss = total_loss + lambda_task_contrastive * loss_dict['task_contrastive_loss']
        return total_loss