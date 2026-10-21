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
                 task_contrastive_temp=0.07):
        """
        Args:
            ... (继承基类参数) ...
        """
        # 初始化基类
        super().__init__(encoder, tokenizer, dag_per_task, hidden_dim,
                        num_heads, num_layers, kappa, label_details_path,
                        use_token_sim, use_bert_token_sim, 
                        use_enhancement, enhancement_config)
        
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
        
        # 2. 任务类型映射加载 (含自动反转逻辑)
        self.task_type_map = {}
        self.ordinal_tasks = {}
        
        if task_type_json and os.path.exists(task_type_json):
            try:
                with open(task_type_json, 'r') as f:
                    raw_data = json.load(f)
                # 检查数据结构：如果是 List 说明是 {"type": [task_list]} 格式，需要反转
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
                num_tasks=36,
                embed_dim=self.embed_dim,
                adapter_dim=adapter_dim,
                use_adapter=self.use_task_adapter
            )
            print(f"[ImprovedCausalGNN] 已启用任务嵌入{'与Adapter' if self.use_task_adapter else ''}")
    
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
        
        # 1. 智能编码器选择
        task_type = self.task_type_map.get(task_name, 'semantic')
        use_hybrid = False
        
        # [FIX] 混合任务也启用 Hybrid
        if self.use_hybrid_encoder and task_type in ['numerical', 'mixed']:
            if numeric_features is not None and feature_mask is not None:
                nf_sum = feature_mask.float().sum().item() if isinstance(feature_mask, torch.Tensor) else sum(feature_mask)
                if nf_sum >= 1: 
                    use_hybrid = True
        
        if self.use_hybrid_encoder and self.use_symbolic_encoder and task_type == 'symbolic':
            use_hybrid = True

        if use_hybrid:
            text_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            
            # [CRITICAL FIX for Error 1] 修复 feature_mask 维度问题
            # Encoder 期望 feature_mask 是 1D (num_features,)，但 DataLoader 给了 2D (batch, num_features)
            feature_mask_arg = feature_mask
            if feature_mask is not None and isinstance(feature_mask, torch.Tensor) and feature_mask.dim() > 1:
                feature_mask_arg = feature_mask[0] # 取第一行，因为同个 batch 任务相同 mask 相同
            
            sample_embed = self.hybrid_encoder(
                text_input=text_input,
                numeric_features=numeric_features,
                feature_mask=feature_mask_arg # 传递修复后的 mask
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
        
        # 4. 获取DAG
        dag = self.dag_per_task.get(task_name, None)
        if dag is None:
            raise ValueError(f"Task {task_name} not found in dag_per_task")
        
        # 5. DAG Text Embed
        if any(n.text_embed is None for n in dag.nodes):
            texts = [n.text for n in dag.nodes]
            with torch.no_grad():
                enc_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
                enc_outputs = self.encoder(**enc_inputs)
                node_emb = enc_outputs.pooler_output
            for i, n in enumerate(dag.nodes):
                n.text_embed = node_emb[i].detach()
        
        node_emb = torch.stack([n.text_embed for n in dag.nodes], dim=0)
        
        # 增强模块
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

        s_v = torch.matmul(sample_embed_norm, node_emb_norm.t())
        
        node_logits_list = []
        attn_entropies = []
        for b in range(batch_size):
            s_v_b = s_v[b]
            node_logits_b, attn_e = self.gnn(dag, s_v_b)
            node_logits_list.append(node_logits_b)
            attn_entropies.extend(attn_e)
        node_logits_batch = torch.stack(node_logits_list, dim=0)
        
        # 8. Zero-Shot Readout
        task_topk_map = getattr(self, 'zs_task_topk_map', None)
        k = 6
        if isinstance(task_topk_map, dict) and task_name in task_topk_map:
            val = task_topk_map[task_name]
            if isinstance(val, float) or (isinstance(val, str) and '.' in val):
                k = max(1, int(len(dag.nodes) * float(val)))
            else:
                k = int(val)
        else:
            k = max(1, int(len(dag.nodes) * 0.5))
        k = min(k, len(dag.nodes))
        k = max(1, k)
        
        topk_values, topk_indices = torch.topk(node_logits_batch, k, dim=1)
        attn_weights = F.softmax(topk_values, dim=1)
        topk_node_emb = torch.stack([node_emb[topk_indices[b]] for b in range(batch_size)], dim=0)
        gnn_enhanced_emb = torch.sum(attn_weights.unsqueeze(-1) * topk_node_emb, dim=1)
        
        fusion_weight = 0.5
        if hasattr(self, 'zs_task_mix_alpha_map') and task_name in self.zs_task_mix_alpha_map:
            fusion_weight = self.zs_task_mix_alpha_map[task_name]
        
        fused_sample_emb = (1 - fusion_weight) * sample_embed + fusion_weight * gnn_enhanced_emb
        fused_sample_emb_norm = F.normalize(fused_sample_emb, dim=-1)
        
        # Label Proto
        label_details = self.label_details.get(task_name, {}) if self.label_details else {}
        label_names = dag.labels
        label_embeds = []
        for lbl in label_names:
            if label_details and lbl in label_details:
                proto_texts = label_details[lbl] if isinstance(label_details[lbl], list) else [label_details[lbl]]
            else:
                proto_texts = [lbl]
            proto_inputs = self.tokenizer(proto_texts, padding=True, truncation=True, return_tensors='pt')
            proto_inputs = {k: v.to(device) for k, v in proto_inputs.items()}
            proto_outputs = self.encoder(**proto_inputs)
            label_embeds.append(proto_outputs.pooler_output.mean(dim=0))
        
        label_embeds = torch.stack(label_embeds)
        label_embeds_norm = F.normalize(label_embeds, dim=-1)
        logits = torch.matmul(fused_sample_emb_norm, label_embeds_norm.t())
        
        tau = 0.5
        if hasattr(self, 'zs_task_tau_map') and isinstance(self.zs_task_tau_map, dict):
            tau = float(self.zs_task_tau_map.get(task_name, 0.5))
        logits = logits / tau
        
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
            
            # 🟢 [STRATEGY CHANGE] 暂时禁用 Ordinal Loss
            # 原因：car-evaluation 崩溃证明 Cosine Similarity Logits 不适合直接用于 Ordinal Loss。
            # 为了恢复 70%+ 的准确率，我们回退到 Cross Entropy。
            
            # if self.use_ordinal_loss and task_name in self.ordinal_tasks:
            #     aux['use_ordinal_loss'] = True
            #     aux['ordinal_task_info'] = self.ordinal_tasks[task_name]
            #     aux['label_names'] = label_names
            # else:
            #     aux['use_ordinal_loss'] = False
            
            aux['use_ordinal_loss'] = False # 强制关闭
        
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