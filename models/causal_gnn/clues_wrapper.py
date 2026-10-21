import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
from collections import defaultdict

from .model import CausalGNNModel
from .representation_enhancement import (
    RepresentationEnhancementModule,
    InfoNCELoss,
    DAGTopologyLoss,
    EdgeSmoothnessLoss,
    MultiLevelContrastiveLoss
)


class CausalGNNClues(nn.Module):
    def __init__(self, encoder, tokenizer, dag_per_task, hidden_dim=256,
                 num_heads=4, num_layers=2, kappa=1.0, label_details_path=None,
                 use_token_sim=False, use_bert_token_sim=False, 
                 use_enhancement=False, enhancement_config=None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.dag_per_task = dag_per_task
        # infer embedding dim from encoder
        self.embed_dim = encoder.config.hidden_size
        self.gnn = CausalGNNModel(embed_dim=self.embed_dim,
                                  hidden_dim=hidden_dim,
                                  num_heads=num_heads,
                                  num_layers=num_layers,
                                  kappa=kappa)
        self.is_causal_gnn = True
        self.use_token_sim = use_token_sim  # 是否使用Token-level相似度(字符串匹配)
        self.use_bert_token_sim = use_bert_token_sim  # 是否使用BERT token嵌入匹配
        # 任务类型、门控与温标、阶段日程默认
        self.task_type_map = {}  # {task_name: semantic|numeric|symbolic}
        self.train_progress = 0.0
        self.gating_g0 = 0.8
        self.gating_gmin = 0.3
        self.topbeta_ratio = 0.7
        self.sim_temp_tok = 0.8
        self.sim_temp_cls = 1.0
        self.phase1_delta_topk = 2
        self.phase1_delta_tau = 0.05
        self.phase1_delta_mix = 0.05
        
        # 表示增强模块
        self.use_enhancement = use_enhancement
        if self.use_enhancement:
            self.enhancement = RepresentationEnhancementModule(
                embed_dim=self.embed_dim,
                config=enhancement_config
            )
            # DAG和多层对比学习损失
            self.dag_topology_loss = DAGTopologyLoss()
            self.edge_smoothness_loss = EdgeSmoothnessLoss()
            self.info_nce_loss = InfoNCELoss(temperature=0.5)
            self.multi_nce_loss = MultiLevelContrastiveLoss(temperature=0.5)
            print(f"[CausalGNN] 已启用表示增强模块")
        
        # 加载标签信息化原型（可选）
        self.label_details = None
        if label_details_path:
            import json
            import os
            if os.path.exists(label_details_path):
                with open(label_details_path, 'r') as f:
                    self.label_details = json.load(f)
                print(f"[CausalGNN] 已加载标签信息化原型: {label_details_path}")
                # 可选：多原型去冗余与判别词强化（基于原型集合内的统计）
                try:
                    self._refine_label_details()
                except Exception:
                    pass
                if self.use_bert_token_sim:
                    print(f"[CausalGNN] 已启用BERT token嵌入相似度计算")
                elif self.use_token_sim:
                    print(f"[CausalGNN] 已启用Token-level相似度计算(字符串匹配)")

    def set_progress(self, progress: float):
        # progress in [0,1]
        self.train_progress = float(max(0.0, min(1.0, progress)))

    def _refine_label_details(self):
        """多原型去冗余与判别词强化（轻量版）
        - 冗余判定：原型字符串分词为集合，Jaccard>阈值则去重
        - 判别词：按标签内词频与其他标签词频比值排序，追加Top-K到原型文本末尾
        阈值/数量由 self.proto_dedupe_thresh / self.proto_disc_k 控制（若存在）
        """
        dedupe_thresh = float(getattr(self, 'proto_dedupe_thresh', 0.85))
        disc_k = int(getattr(self, 'proto_disc_k', 3))
        for task_name, lbl_map in list(self.label_details.items()):
            # 1) 统计全任务的词频
            word_df = {}
            label_word_tf = {}
            all_labels = list(lbl_map.keys())
            for lbl in all_labels:
                protos = lbl_map[lbl]
                if not isinstance(protos, list):
                    protos = [protos]
                tf = {}
                seen_words = set()
                for text in protos:
                    tokens = set(re.findall(r'\b\w+\b', str(text).lower()))
                    for w in tokens:
                        tf[w] = tf.get(w, 0) + 1
                        if w not in seen_words:
                            word_df[w] = word_df.get(w, 0) + 1
                            seen_words.add(w)
                label_word_tf[lbl] = tf
            # 2) 每标签：去冗余 & 判别词
            for lbl in all_labels:
                protos = lbl_map[lbl]
                if not isinstance(protos, list):
                    protos = [protos]
                # 去冗余（Jaccard）
                kept = []
                sigs = []
                for text in protos:
                    toks = set(re.findall(r'\b\w+\b', str(text).lower()))
                    is_dup = False
                    for prev in sigs:
                        inter = len(toks & prev)
                        union = max(1, len(toks | prev))
                        j = inter / union
                        if j >= dedupe_thresh:
                            is_dup = True
                            break
                    if not is_dup:
                        kept.append(text)
                        sigs.append(toks)
                # 判别词（基于TF/DF比）
                tf = label_word_tf.get(lbl, {})
                scores = []
                for w, f in tf.items():
                    df = max(1, word_df.get(w, 1))
                    scores.append((w, f / df))
                scores.sort(key=lambda x: x[1], reverse=True)
                top_words = [w for w, _ in scores[:disc_k]]
                if top_words:
                    aug_suffix = " including: " + ", ".join(top_words)
                    kept = [str(k) + aug_suffix for k in kept]
                lbl_map[lbl] = kept
            self.label_details[task_name] = lbl_map

    def encode_text(self, input_ids, attention_mask, token_type_ids):
        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]
    
    def extract_key_tokens(self, text):
        """
        从文本中提取关键token,用于Token-level对齐
        去除标点、停用词,保留有意义的关键词
        """
        # 简化的停用词列表
        stop_words = {'the', 'is', 'are', 'a', 'an', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'at', 'by', 'with'}
        # 分词并清理
        tokens = re.findall(r'\b\w+\b', text.lower())
        # 过滤停用词和单字符
        key_tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return key_tokens
    
    def compute_token_level_similarity(self, sample_texts, node_texts, device):
        """
        计算Token-level对齐的相似度
        只计算关键Token的相似度,加入位置感知
        
        Args:
            sample_texts: List[str], 样本文本列表 [B]
            node_texts: List[str], 节点文本列表 [N]
            device: torch.device
        
        Returns:
            sims: torch.Tensor [B, N], 样本-节点相似度矩阵
        """
        B = len(sample_texts)
        N = len(node_texts)
        sims = torch.zeros(B, N, device=device)
        
        for b, sample_text in enumerate(sample_texts):
            # 提取样本的关键token
            sample_tokens = self.extract_key_tokens(sample_text)
            if len(sample_tokens) == 0:
                continue
            
            # 构建样本的token→value映射(位置感知)
            # 例如: "top_middle_square" -> {"top": "middle", "middle": "square"}
            sample_token_context = defaultdict(list)
            for i, token in enumerate(sample_tokens):
                # 记录上下文token
                if i > 0:
                    sample_token_context[token].append(sample_tokens[i-1])
                if i < len(sample_tokens) - 1:
                    sample_token_context[token].append(sample_tokens[i+1])
            
            for n, node_text in enumerate(node_texts):
                # 提取节点的关键token
                node_tokens = self.extract_key_tokens(node_text)
                if len(node_tokens) == 0:
                    continue
                
                # 构建节点的token→value映射
                node_token_context = defaultdict(list)
                for i, token in enumerate(node_tokens):
                    if i > 0:
                        node_token_context[token].append(node_tokens[i-1])
                    if i < len(node_tokens) - 1:
                        node_token_context[token].append(node_tokens[i+1])
                
                # 计算匹配分数
                match_score = 0.0
                matched_tokens = 0
                
                for token in sample_tokens:
                    if token in node_tokens:
                        # 基础匹配: +1
                        match_score += 1.0
                        matched_tokens += 1
                        
                        # 位置感知加分: 如果上下文token也匹配,再 +0.5
                        sample_context = set(sample_token_context[token])
                        node_context = set(node_token_context[token])
                        context_overlap = len(sample_context & node_context)
                        if context_overlap > 0:
                            match_score += 0.5 * context_overlap
                
                # 归一化: Jaccard-like相似度
                total_tokens = len(set(sample_tokens) | set(node_tokens))
                if total_tokens > 0:
                    sims[b, n] = match_score / total_tokens
        
        return sims
    
    def compute_bert_token_similarity(self, sample_texts, node_texts, device):
        """
        使用BERT token-level嵌入计算相似度
        对每对(sample, node)，计算最优token对齐
        
        Args:
            sample_texts: List[str], 样本文本列表 [B]
            node_texts: List[str], 节点文本列表 [N]
            device: torch.device
        
        Returns:
            sims: torch.Tensor [B, N], 样本-节点相似度矩阵
        """
        B = len(sample_texts)
        N = len(node_texts)
        sims = torch.zeros(B, N, device=device)
        
        # 编码样本（获取所有token嵌入，不只是CLS）
        sample_inputs = self.tokenizer(sample_texts, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            sample_outputs = self.encoder(**sample_inputs)
            sample_hidden = sample_outputs.last_hidden_state  # [B, L1, D]
        
        # 编码节点
        node_inputs = self.tokenizer(node_texts, padding=True, truncation=True,
                                     max_length=512, return_tensors='pt').to(device)
        with torch.no_grad():
            node_outputs = self.encoder(**node_inputs)
            node_hidden = node_outputs.last_hidden_state  # [N, L2, D]
        
        # 对每对(sample, node)，计算最优token对齐
        for b in range(B):
            sample_tokens = sample_hidden[b]  # [L1, D]
            sample_mask = sample_inputs['attention_mask'][b]  # [L1]
            sample_tokens = sample_tokens[sample_mask == 1]  # 去除padding，[L1_valid, D]
            
            # 跳过[CLS]和[SEP]
            if sample_tokens.size(0) > 2:
                sample_tokens = sample_tokens[1:-1]  # 去除[CLS]和[SEP]
            
            if sample_tokens.size(0) == 0:
                continue
            
            for n in range(N):
                node_tokens = node_hidden[n]  # [L2, D]
                node_mask = node_inputs['attention_mask'][n]  # [L2]
                node_tokens = node_tokens[node_mask == 1]  # 去除padding，[L2_valid, D]
                
                # 跳过[CLS]和[SEP]
                if node_tokens.size(0) > 2:
                    node_tokens = node_tokens[1:-1]
                
                if node_tokens.size(0) == 0:
                    continue
                
                # 计算token-token余弦相似度矩阵
                sample_norm = sample_tokens / (sample_tokens.norm(dim=-1, keepdim=True) + 1e-8)
                node_norm = node_tokens / (node_tokens.norm(dim=-1, keepdim=True) + 1e-8)
                sim_matrix = torch.matmul(sample_norm, node_norm.t())  # [L1, L2]
                
                # 最优对齐：对每个sample token取最大相似度，再对Top-β进行平均
                max_sims, _ = sim_matrix.max(dim=1)  # [L1]
                if max_sims.numel() > 0:
                    beta = float(getattr(self, 'topbeta_ratio', 0.7))
                    k = max(1, int(math.ceil(max_sims.numel() * beta)))
                    topk_vals, _ = torch.topk(max_sims, k)
                    sims[b, n] = topk_vals.mean().item()
        
        return sims

    def forward(self, input_ids, attention_mask, token_type_ids,
                task_name,
                target_text_matrix,
                entail_indicators,
                guideline_inputs=None,
                **kwargs):
        device = input_ids.device
        dag = self.dag_per_task[task_name]
        dag = dag.to_device(device)

        # sample embedding
        sample_embed = self.encode_text(input_ids, attention_mask, token_type_ids)  # [B, D]

        # 确保 DAG 节点有 text_embed（手工DAG载入时为None，需要即刻编码）
        if any(n.text_embed is None for n in dag.nodes):
            texts = [n.text for n in dag.nodes]
            with torch.no_grad():
                enc_inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
                enc_outputs = self.encoder(**enc_inputs)
                if hasattr(enc_outputs, 'pooler_output') and enc_outputs.pooler_output is not None:
                    node_emb = enc_outputs.pooler_output
                else:
                    node_emb = enc_outputs.last_hidden_state[:, 0]
            for i, n in enumerate(dag.nodes):
                n.text_embed = node_emb[i]

        # compute s_v per node via cosine similarity between sample and node text embeds
        node_texts = [n.text for n in dag.nodes]
        node_emb = torch.stack([n.text_embed for n in dag.nodes], dim=0)  # [N, D]
        
        # ========== 表示增强模块 ==========
        enhancement_losses = {}
        if self.use_enhancement:
            # 1. 白化变换(应用于样本和节点嵌入)
            sample_embed_enhanced = self.enhancement.apply_whitening(
                sample_embed, update_stats=self.training
            )
            node_emb_enhanced = self.enhancement.apply_whitening(
                node_emb, update_stats=self.training
            )
            
            # 2. 随机噪声增强(仅训练时)
            if self.training:
                sample_embed_enhanced = self.enhancement.apply_noise(
                    sample_embed_enhanced, training=True
                )
                node_emb_enhanced = self.enhancement.apply_noise(
                    node_emb_enhanced, training=True
                )
            
            # 3. 计算各向同性损失
            if self.training:
                iso_loss_sample = self.enhancement.compute_isotropy_loss(sample_embed_enhanced)
                iso_loss_node = self.enhancement.compute_isotropy_loss(node_emb_enhanced)
                enhancement_losses['isotropy'] = iso_loss_sample + iso_loss_node
            
            # 使用增强后的嵌入
            sample_embed_for_sim = sample_embed_enhanced
            node_emb_for_sim = node_emb_enhanced
        else:
            sample_embed_for_sim = sample_embed
            node_emb_for_sim = node_emb
        
        # 根据配置选择相似度计算方式
        if self.use_bert_token_sim:
            # BERT token嵌入对齐相似度
            sample_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            sims_tok = self.compute_bert_token_similarity(sample_texts, node_texts, device)  # [B, N]
            # 同时计算CLS相似度用于对比
            node_norm = node_emb_for_sim / (node_emb_for_sim.norm(dim=-1, keepdim=True) + 1e-8)
            samp_norm = sample_embed_for_sim / (sample_embed_for_sim.norm(dim=-1, keepdim=True) + 1e-8)
            sims_cls = torch.matmul(samp_norm, node_norm.t())  # [B, N]
            # 温标与门控（Token/CLS）+ Top-β 筛选
            tau_tok = max(1e-6, float(getattr(self, 'sim_temp_tok', 0.8)))
            tau_cls = max(1e-6, float(getattr(self, 'sim_temp_cls', 1.0)))
            sims_tok = sims_tok / tau_tok
            sims_cls = sims_cls / tau_cls
            # 任务类型分支，调整门控范围
            task_type = (self.task_type_map.get(task_name) or '').lower()
            if task_type in ('numeric', 'symbolic', 'rules', 'rule', 'symbolic-rule'):
                g0 = float(getattr(self, 'gating_g0', 0.4))
                gmin = float(getattr(self, 'gating_gmin', 0.1))
            else:
                g0 = float(getattr(self, 'gating_g0', 0.8))
                gmin = float(getattr(self, 'gating_gmin', 0.3))
            g = max(gmin, g0 - (g0 - gmin) * float(getattr(self, 'train_progress', 0.0)))
            sims = g * sims_tok + (1.0 - g) * sims_cls
        elif self.use_token_sim:
            # Token-level对齐相似度(字符串匹配)
            sample_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            sims_tok = self.compute_token_level_similarity(sample_texts, node_texts, device)  # [B, N]
            # 同时计算CLS相似度用于对比
            node_norm = node_emb_for_sim / (node_emb_for_sim.norm(dim=-1, keepdim=True) + 1e-8)
            samp_norm = sample_embed_for_sim / (sample_embed_for_sim.norm(dim=-1, keepdim=True) + 1e-8)
            sims_cls = torch.matmul(samp_norm, node_norm.t())  # [B, N]
            tau_tok = max(1e-6, float(getattr(self, 'sim_temp_tok', 0.8)))
            tau_cls = max(1e-6, float(getattr(self, 'sim_temp_cls', 1.0)))
            sims_tok = sims_tok / tau_tok
            sims_cls = sims_cls / tau_cls
            task_type = (self.task_type_map.get(task_name) or '').lower()
            if task_type in ('numeric', 'symbolic', 'rules', 'rule', 'symbolic-rule'):
                g0 = float(getattr(self, 'gating_g0', 0.4))
                gmin = float(getattr(self, 'gating_gmin', 0.1))
            else:
                g0 = float(getattr(self, 'gating_g0', 0.8))
                gmin = float(getattr(self, 'gating_gmin', 0.3))
            g = max(gmin, g0 - (g0 - gmin) * float(getattr(self, 'train_progress', 0.0)))
            sims = g * sims_tok + (1.0 - g) * sims_cls
        else:
            # 原始CLS相似度
            node_norm = node_emb_for_sim / (node_emb_for_sim.norm(dim=-1, keepdim=True) + 1e-8)
            samp_norm = sample_embed_for_sim / (sample_embed_for_sim.norm(dim=-1, keepdim=True) + 1e-8)
            sims = torch.matmul(samp_norm, node_norm.t())  # [B, N]
            sims_cls = sims  # 保持一致
        
        # ========== 相似度增强 ==========
        if self.use_enhancement:
            # 4. 温度标定
            sims = self.enhancement.apply_temperature_scaling(sims)
            
            # 5. 局部标度归一(可选,适用于CLS相似度)
            if not self.use_bert_token_sim and not self.use_token_sim:
                sims = self.enhancement.apply_local_calibration(sims)
            
            # 6. 计算相似度熵损失
            if self.training:
                sim_ent_loss = self.enhancement.compute_similarity_entropy_loss(
                    node_emb_for_sim, sample_embed_for_sim
                )
                enhancement_losses['similarity_entropy'] = sim_ent_loss

        B = sims.size(0)
        node_logits_list = []
        attn_entropies = []
        for b in range(B):
            s_v_b = sims[b]
            # 子图TopK掩码（可选）
            sub_topk = int(getattr(self, 'gnn_subgraph_topk', 0))
            if sub_topk and sub_topk > 0 and s_v_b.numel() > sub_topk:
                vals, idx = torch.topk(s_v_b, k=sub_topk)
                mask = torch.zeros_like(s_v_b, dtype=torch.bool)
                mask[idx] = True
                s_v_b = torch.where(mask, s_v_b, torch.zeros_like(s_v_b))
            node_logits_b, attn_e = self.gnn(dag, s_v_b)
            node_logits_list.append(node_logits_b)
            attn_entropies.extend(attn_e)
        node_logits_batch = torch.stack(node_logits_list, dim=0)  # [B, N]

        # 优先使用 DAG 内置的标签映射
        if getattr(dag, 'labels', None) and getattr(dag, 'label_nodes', None) and len(dag.labels) > 0:
            label_to_ids = [dag.label_nodes.get(lbl, []) for lbl in dag.labels]
            scores_batch = []  # [B, L]
            for node_ids in label_to_ids:
                if len(node_ids) == 0:
                    # 若该标签未映射任何节点，则给0分
                    scores_batch.append(torch.zeros(node_logits_batch.size(0), device=node_logits_batch.device))
                else:
                    mask = torch.zeros(node_logits_batch.size(-1), dtype=torch.bool, device=node_logits_batch.device)
                    for nid in node_ids:
                        if nid < mask.numel():
                            mask[nid] = True
                    scores_batch.append(node_logits_batch[:, mask].sum(dim=-1))
            scores = torch.stack(scores_batch, dim=-1)  # [B, L]
            return scores, {
                "attn_entropies": attn_entropies, 
                "node_logits": node_logits_batch,
                "enhancement_losses": enhancement_losses
            }

        # 若未提供 label_nodes，优先根据配置选择零样本读出
        zs_readout = getattr(self, 'zs_readout', 'guideline')
        zs_topk = int(getattr(self, 'zs_topk', 8))
        zs_tau = float(getattr(self, 'zs_tau', 0.5))

        # 1) 标签原型读出（不依赖 guideline）
        if zs_readout == 'label-proto' and getattr(dag, 'labels', None) and len(dag.labels) > 0:
            # 编码标签文本（支持信息化多原型）
            with torch.no_grad():
                # 检查是否有信息化原型
                if self.label_details and task_name in self.label_details:
                    # 使用信息化多原型
                    all_label_texts = []
                    label_proto_counts = []  # 记录每个标签有几条原型
                    for lbl in dag.labels:
                        if lbl in self.label_details[task_name]:
                            proto_texts = self.label_details[task_name][lbl]
                            if not isinstance(proto_texts, list):
                                proto_texts = [proto_texts]
                            all_label_texts.extend(proto_texts)
                            label_proto_counts.append(len(proto_texts))
                        else:
                            # 回退到默认原型
                            all_label_texts.append(f"label: {lbl}")
                            label_proto_counts.append(1)
                    # 编码所有原型
                    l_inputs = self.tokenizer(all_label_texts, padding=True, truncation=True, return_tensors='pt').to(device)
                    l_outputs = self.encoder(**l_inputs)
                    if hasattr(l_outputs, 'pooler_output') and l_outputs.pooler_output is not None:
                        all_proto_emb = l_outputs.pooler_output  # [总原型数, D]
                    else:
                        all_proto_emb = l_outputs.last_hidden_state[:, 0]
                    # 对每个标签的多个原型取平均（或最大值）
                    label_emb = []
                    start_idx = 0
                    for count in label_proto_counts:
                        proto_emb_group = all_proto_emb[start_idx:start_idx+count]
                        # 取平均作为该标签的最终嵌入
                        label_emb.append(proto_emb_group.mean(dim=0))
                        start_idx += count
                    label_emb = torch.stack(label_emb, dim=0)  # [L, D]
                else:
                    # 默认原型
                    label_texts = [f"label: {lbl}" for lbl in dag.labels]
                    l_inputs = self.tokenizer(label_texts, padding=True, truncation=True, return_tensors='pt').to(device)
                    l_outputs = self.encoder(**l_inputs)
                    if hasattr(l_outputs, 'pooler_output') and l_outputs.pooler_output is not None:
                        label_emb = l_outputs.pooler_output  # [L, D]
                    else:
                        label_emb = l_outputs.last_hidden_state[:, 0]
            # 计算节点-标签相似度 S: [N, L]
            if self.use_bert_token_sim:
                # BERT token嵌入对齐相似度(节点文本 vs 标签原型文本)
                S_bert = torch.zeros(node_emb.size(0), label_emb.size(0), device=device)
                for j, lbl in enumerate(dag.labels):
                    # 获取该标签的所有原型文本
                    if self.label_details and task_name in self.label_details and lbl in self.label_details[task_name]:
                        proto_texts = self.label_details[task_name][lbl]
                        if not isinstance(proto_texts, list):
                            proto_texts = [proto_texts]
                    else:
                        proto_texts = [f"label: {lbl}"]
                    
                    # 对每个原型计算与所有节点的BERT token嵌入相似度,取平均值
                    proto_sims = []
                    for proto_text in proto_texts:
                        sim_col = self.compute_bert_token_similarity([proto_text], node_texts, device).squeeze(0)  # [N]
                        proto_sims.append(sim_col)
                    # 取平均
                    S_bert[:, j] = torch.stack(proto_sims, dim=0).mean(dim=0)
                
                S = S_bert
                # 同时计算CLS相似度用于对比
                node_emb_norm = node_emb / (node_emb.norm(dim=-1, keepdim=True) + 1e-8)
                label_emb_norm = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-8)
                S_cls = torch.matmul(node_emb_norm, label_emb_norm.t())
            elif self.use_token_sim:
                # Token-level对齐相似度(节点文本 vs 标签原型文本, 字符串匹配)
                S_token = torch.zeros(node_emb.size(0), label_emb.size(0), device=device)
                for j, lbl in enumerate(dag.labels):
                    # 获取该标签的所有原型文本
                    if self.label_details and task_name in self.label_details and lbl in self.label_details[task_name]:
                        proto_texts = self.label_details[task_name][lbl]
                        if not isinstance(proto_texts, list):
                            proto_texts = [proto_texts]
                    else:
                        proto_texts = [f"label: {lbl}"]
                    
                    # 对每个原型计算与所有节点的Token-level相似度,取最大值或平均值
                    proto_sims = []
                    for proto_text in proto_texts:
                        sim_col = self.compute_token_level_similarity([proto_text], node_texts, device).squeeze(0)  # [N]
                        proto_sims.append(sim_col)
                    # 取平均(也可改为max)
                    S_token[:, j] = torch.stack(proto_sims, dim=0).mean(dim=0)
                
                S = S_token
                # 同时计算CLS相似度用于对比
                node_emb_norm = node_emb / (node_emb.norm(dim=-1, keepdim=True) + 1e-8)
                label_emb_norm = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-8)
                S_cls = torch.matmul(node_emb_norm, label_emb_norm.t())
            else:
                # 原始CLS余弦相似度
                node_emb_norm = node_emb / (node_emb.norm(dim=-1, keepdim=True) + 1e-8)
                label_emb_norm = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-8)
                S = torch.matmul(node_emb_norm, label_emb_norm.t())
                S_cls = S  # 保持一致
            # 生成权重 W: [N, L]
            # 任务自适应 TopK / 温度覆盖
            task_topk_map = getattr(self, 'zs_task_topk_map', None)
            task_tau_map = getattr(self, 'zs_task_tau_map', None)
            adapt = bool(getattr(self, 'zs_adapt', False))
            adapt_pct = float(getattr(self, 'zs_adapt_topk_pct', 0.15))
            # 两阶段日程与任务级覆盖
            task_topk = zs_topk
            task_tau = zs_tau
            if isinstance(task_topk_map, dict) and task_name in task_topk_map:
                task_topk = int(task_topk_map[task_name])
            if isinstance(task_tau_map, dict) and task_name in task_tau_map:
                task_tau = float(task_tau_map[task_name])
            if adapt:
                task_topk = max(1, int(node_emb.size(0) * adapt_pct))
            # 两阶段放宽（前半程更平）
            if float(getattr(self, 'train_progress', 0.0)) < 0.5:
                task_topk = min(node_emb.size(0), int(task_topk + int(getattr(self, 'phase1_delta_topk', 2))))
                task_tau = float(task_tau + float(getattr(self, 'phase1_delta_tau', 0.05)))

            if task_topk and task_topk > 0:
                # TopK 掩码
                topk_vals, topk_idx = torch.topk(S, k=min(task_topk, S.size(0)), dim=0)
                W = torch.zeros_like(S)
                for j in range(S.size(1)):
                    idx = topk_idx[:, j]
                    W[idx, j] = topk_vals[:, j]
                # 列归一化
                W = W / (W.sum(dim=0, keepdim=True) + 1e-8)
            else:
                W = torch.softmax(S / max(1e-6, task_tau), dim=0)
            # 读出：可选非负化与样本证据融合
            node_vals = node_logits_batch
            if bool(getattr(self, 'zs_nonneg', False)):
                node_vals = F.softplus(node_vals)
            scores_proto = torch.matmul(node_vals, W)  # [B, L]

            # 与 guideline 读出混合（可选，提升稳健性）
            # 支持任务级 mix_alpha 覆盖
            task_mix_alpha_map = getattr(self, 'zs_task_mix_alpha_map', None)
            mix_alpha = float(getattr(self, 'zs_mix_alpha', 0.0))
            if isinstance(task_mix_alpha_map, dict) and task_name in task_mix_alpha_map:
                mix_alpha = float(task_mix_alpha_map[task_name])
            if float(getattr(self, 'train_progress', 0.0)) < 0.5:
                mix_alpha = min(0.65, mix_alpha + float(getattr(self, 'phase1_delta_mix', 0.05)))
            if mix_alpha > 0 and 'guideline_input_ids' in (guideline_inputs or {}):
                # 复用下方 guideline 路径，但只构造掩码并求和
                # 简化：用 sim_matrix TopK 生成 mask 并聚合
                with torch.no_grad():
                    g_inputs = self.tokenizer.batch_decode(guideline_inputs['guideline_input_ids'], skip_special_tokens=True)
                    g_enc = self.tokenizer(g_inputs, padding=True, truncation=True, return_tensors='pt').to(device)
                    g_outputs = self.encoder(**g_enc)
                    g_emb = g_outputs.pooler_output if hasattr(g_outputs, 'pooler_output') and g_outputs.pooler_output is not None else g_outputs.last_hidden_state[:, 0]
                    node_emb_norm = node_emb / (node_emb.norm(dim=-1, keepdim=True) + 1e-8)
                    g_emb_norm = g_emb / (g_emb.norm(dim=-1, keepdim=True) + 1e-8)
                    sim_matrix = torch.matmul(node_emb_norm, g_emb_norm.t())  # [N, G]
                aligned_matrix = []
                for target_select in target_text_matrix:
                    mask = torch.zeros(node_vals.size(-1), dtype=torch.bool, device=node_vals.device)
                    guideline_indices = target_select.nonzero(as_tuple=False).view(-1)
                    if len(guideline_indices) > 0:
                        relevant_sims = sim_matrix[:, guideline_indices]
                        max_sims, _ = relevant_sims.max(dim=-1)
                        k = max(1, min(int(getattr(self, 'zs_topk', 8)), node_vals.size(-1)))
                        topk = torch.topk(max_sims, k=k)
                        mask[topk.indices] = True
                    else:
                        mask[:] = True
                    aligned_matrix.append(mask)
                scores_guides = []
                for mask in aligned_matrix:
                    scores_guides.append(node_vals[:, mask].sum(dim=-1))
                scores_guides = torch.stack(scores_guides, dim=-1)
                scores = (1.0 - mix_alpha) * scores_proto + mix_alpha * scores_guides
            else:
                scores = scores_proto
            # 可选：导出每任务的标签原型相似度直方图（用于超参选择）
            hist_dir = getattr(self, 'zs_hist_dir', None)
            if hist_dir:
                import os, json
                os.makedirs(hist_dir, exist_ok=True)
                bins = int(getattr(self, 'zs_hist_bins', 30))
                
                # 导出当前使用的相似度(Token-level或CLS)
                hist_per_label = {}
                for j, lbl in enumerate(dag.labels):
                    vals = S[:, j].detach().float().cpu().numpy().tolist()
                    hist_per_label[str(lbl)] = vals
                
                # 计算统计信息
                S_mean = S.mean().item()
                S_std = S.std().item()
                
                # 保存主要相似度
                if self.use_bert_token_sim:
                    sim_type = "bert_token"
                    sim_desc = "BERT token嵌入"
                elif self.use_token_sim:
                    sim_type = "token"
                    sim_desc = "Token-level"
                else:
                    sim_type = "cls"
                    sim_desc = "CLS"
                
                with open(os.path.join(hist_dir, f"proto_sims_{sim_type}_{task_name}.json"), 'w') as f:
                    json.dump({
                        "labels": dag.labels,
                        "sims": hist_per_label,
                        "stats": {"mean": S_mean, "std": S_std},
                        "sim_type": sim_type
                    }, f, ensure_ascii=False, indent=2)
                
                # 写入统计日志文件
                log_file = os.path.join(hist_dir, "similarity_stats.log")
                with open(log_file, 'a') as f:
                    f.write(f"[{task_name}] 相似度统计 ({sim_desc}): mean={S_mean:.4f}, std={S_std:.6f}\n")
                
                # 如果使用BERT token或Token-level,同时导出CLS相似度用于对比
                if self.use_bert_token_sim or self.use_token_sim:
                    hist_per_label_cls = {}
                    for j, lbl in enumerate(dag.labels):
                        vals = S_cls[:, j].detach().float().cpu().numpy().tolist()
                        hist_per_label_cls[str(lbl)] = vals
                    
                    S_cls_mean = S_cls.mean().item()
                    S_cls_std = S_cls.std().item()
                    
                    with open(os.path.join(hist_dir, f"proto_sims_cls_{task_name}.json"), 'w') as f:
                        json.dump({
                            "labels": dag.labels,
                            "sims": hist_per_label_cls,
                            "stats": {"mean": S_cls_mean, "std": S_cls_std},
                            "sim_type": "cls"
                        }, f, ensure_ascii=False, indent=2)
                    
                    # 写入CLS对比统计到日志文件
                    with open(log_file, 'a') as f:
                        f.write(f"[{task_name}] 相似度统计 (CLS对比): mean={S_cls_mean:.4f}, std={S_cls_std:.6f}\n")
            
            # ========== 计算DAG拓扑和边平滑损失 ==========
            if self.use_enhancement and self.training:
                edge_index, edge_weight = dag.edge_index_and_weight()
                if edge_index is not None and edge_index.numel() > 0:
                    edge_index = edge_index.to(device)
                    edge_weight = edge_weight.to(device) if edge_weight.numel() > 0 else torch.ones(edge_index.size(1), device=device)
                    
                    # DAG拓扑一致性损失
                    dag_topo_loss = self.dag_topology_loss(node_emb_for_sim, edge_index, edge_weight)
                    enhancement_losses['dag_topology'] = dag_topo_loss
                    
                    # 边平滑正则损失
                    edge_smooth_loss = self.edge_smoothness_loss(node_emb_for_sim, edge_index, edge_weight)
                    enhancement_losses['edge_smoothness'] = edge_smooth_loss

            return scores, {
                "attn_entropies": attn_entropies, 
                "node_logits": node_logits_batch,
                "enhancement_losses": enhancement_losses
            }

        # —— 对齐 target_text_matrix 到节点数 N ——（仅在无DAG映射时使用）
        N = node_logits_batch.shape[-1]
        aligned_matrix = []
        
        if zs_readout == 'guideline' and guideline_inputs is not None and 'guideline_input_ids' in guideline_inputs:
            # 改进：使用嵌入相似度对齐
            with torch.no_grad():
                g_inputs = self.tokenizer.batch_decode(guideline_inputs['guideline_input_ids'], skip_special_tokens=True)
                # 编码guideline文本
                g_enc = self.tokenizer(g_inputs, padding=True, truncation=True, return_tensors='pt').to(device)
                g_outputs = self.encoder(**g_enc)
                if hasattr(g_outputs, 'pooler_output') and g_outputs.pooler_output is not None:
                    g_emb = g_outputs.pooler_output
                else:
                    g_emb = g_outputs.last_hidden_state[:, 0]
                
                # 计算节点与guideline的相似度
                node_emb_norm = node_emb / (node_emb.norm(dim=-1, keepdim=True) + 1e-8)
                g_emb_norm = g_emb / (g_emb.norm(dim=-1, keepdim=True) + 1e-8)
                sim_matrix = torch.matmul(node_emb_norm, g_emb_norm.t())  # [N, G]
            
            # 为每个label构造掩码
            for target_select in target_text_matrix:
                mask = torch.zeros(N, dtype=torch.bool, device=node_logits_batch.device)
                guideline_indices = target_select.nonzero(as_tuple=False).view(-1)
                
                if len(guideline_indices) > 0:
                    # 找到与这些guideline最相似的节点
                    relevant_sims = sim_matrix[:, guideline_indices]  # [N, K]
                    max_sims, _ = relevant_sims.max(dim=-1)  # [N]
                    
                    # 稳健 TopK：固定 K，避免空掩码
                    k = max(1, min(int(getattr(self, 'zs_topk', 8)), N))
                    topk = torch.topk(max_sims, k=k)
                    mask = torch.zeros(N, dtype=torch.bool, device=node_logits_batch.device)
                    mask[topk.indices] = True
                
                aligned_matrix.append(mask)
                
        if not aligned_matrix:
            # 回退策略：直接使用原矩阵或全选
            for target_select in target_text_matrix:
                if target_select.shape[-1] == N:
                    aligned_matrix.append(target_select.to(node_logits_batch.device).bool())
                else:
                    # 全选策略，避免全零掩码
                    mask = torch.ones(N, dtype=torch.bool, device=node_logits_batch.device)
                    aligned_matrix.append(mask)

        # 聚合为 label 分数
        scores_batch = []
        for mask in aligned_matrix:
            scores_batch.append(node_logits_batch[:, mask].sum(dim=-1))  # [B]
        scores = torch.stack(scores_batch, dim=-1)  # [B, L]

        # entail sign
        entail = (entail_indicators * 2 - 1).to(device)
        if entail.shape[-1] != N:
            # 调整或截断到 N
            if entail.shape[-1] > N:
                entail = entail[..., :N]
            else:
                pad = torch.ones(N - entail.shape[-1], device=device, dtype=entail.dtype)
                entail = torch.cat([entail, pad], dim=-1)
        signed_nodes_batch = node_logits_batch * entail
        scores_signed = []
        for mask in aligned_matrix:
            scores_signed.append(signed_nodes_batch[:, mask].sum(dim=-1))  # [B]
        scores = torch.stack(scores_signed, dim=-1)  # [B, L]
        return scores, {
            "attn_entropies": attn_entropies, 
            "node_logits": node_logits_batch,
            "enhancement_losses": enhancement_losses
        }


