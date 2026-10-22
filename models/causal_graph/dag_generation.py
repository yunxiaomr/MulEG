import math
import re
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple


def build_stats_driven_dag(task_name, task_data, tokenizer, encoder, device, edge_keep_ratio=0.3, use_full_formula=True, use_field_level=False, field_weight_mode='uniform'):
    """
    Args:
        task_name: 任务名称
        task_data: 任务数据，包含'train'/'dev'/'test' splits和'target_key'
        tokenizer: BERT tokenizer
        encoder: BERT encoder
        device: 计算设备
        edge_keep_ratio: 保留边的比例
        use_full_formula: 是否使用完整公式（True）或简化版（False）
        use_field_level: 是否使用字段级相似度计算
        field_weight_mode: 字段权重模式 ('uniform', 'info_gain', 'learnable')
    """
    labels = list(task_data.get('target_key', []))
    nodes = [{"id": i, "name": str(lbl), "text": f"label: {lbl}"} for i, lbl in enumerate(labels)]
    if len(nodes) == 0:
        return {"nodes": [], "edges": [], "labels": [], "label_nodes": {}}

    # 与encoder所在设备对齐
    try:
        enc_device = next(encoder.parameters()).device
    except StopIteration:
        enc_device = device

    # 编码标签原型文本
    l_inputs = tokenizer([n["text"] for n in nodes], padding=True, truncation=True, return_tensors='pt')
    l_inputs = {k: v.to(enc_device) for k, v in l_inputs.items()}
    with torch.no_grad():
        outputs = encoder(**l_inputs)
        label_emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
    label_emb = label_emb / (label_emb.norm(dim=-1, keepdim=True) + 1e-8)

    if use_field_level:
        from models.causal_graph.field_similarity import FieldLevelSimilarity
        
        train_samples = task_data.get('train', [])
        if len(train_samples) == 0:
            # 没有训练样本，回退到标签级
            use_field_level = False
        else:
            # 提取字段名
            field_names = [k for k in train_samples[0].keys() if k not in ['lbl', 'target_key']]
            
            # 创建字段级相似度模块
            field_sim_module = FieldLevelSimilarity(
                encoder=encoder,
                tokenizer=tokenizer,
                field_names=field_names,
                weight_mode=field_weight_mode,
                device=enc_device
            )
            
            # 限制样本数量避免OOM
            max_samples = min(1000, len(train_samples))
            sampled_data = train_samples[:max_samples]
            
            # 计算字段级边权矩阵
            sim_ll = field_sim_module.compute_task_edge_weights(
                task_data={'train': sampled_data, 'target_key': labels},
                labels=labels
            )
    
    # 计算边权矩阵（标签级）
    if not use_field_level and use_full_formula:
        #  完整版公式: w_{i→j} = (1/|X_i|) Σ_{x∈X_i} max_{c∈C_j} cos(h_i(x), p_j(c))
        # 由于我们在DAG构建阶段，这里用训练样本作为X_i的代理
        sim_ll = np.zeros((len(labels), len(labels)))
        
        # 获取训练样本（如果有的话）
        train_samples = task_data.get('train', [])
        if len(train_samples) > 0:
            # 限制样本数量避免OOM
            max_samples_per_label = min(100, len(train_samples) // len(labels) + 1)
            
            # 按标签分组样本
            label_to_samples = {lbl: [] for lbl in labels}
            for sample in train_samples:
                sample_label = sample.get('lbl', None)
                if sample_label in label_to_samples and len(label_to_samples[sample_label]) < max_samples_per_label:
                    # 构造样本文本
                    sample_text = "For this item, " + ", ".join([
                        f"{k} is {v}" for k, v in sample.items() 
                        if k not in ['lbl', 'target_key']
                    ]) + "."
                    label_to_samples[sample_label].append(sample_text)
            
            # 计算每对标签间的样本-原型相似度
            for i, label_i in enumerate(labels):
                samples_i = label_to_samples[label_i]
                if len(samples_i) == 0:
                    # 如果没有样本，回退到标签间相似度
                    sim_ll[i] = (label_emb[i:i+1] @ label_emb.t()).detach().cpu().numpy()[0]
                    continue
                
                # 编码样本
                s_inputs = tokenizer(samples_i, padding=True, truncation=True, max_length=512, return_tensors='pt')
                s_inputs = {k: v.to(enc_device) for k, v in s_inputs.items()}
                with torch.no_grad():
                    s_outputs = encoder(**s_inputs)
                    sample_emb = s_outputs.pooler_output if hasattr(s_outputs, 'pooler_output') and s_outputs.pooler_output is not None else s_outputs.last_hidden_state[:, 0]
                sample_emb = sample_emb / (sample_emb.norm(dim=-1, keepdim=True) + 1e-8)
                
                # 对每个目标标签j，计算 avg(max_cos(样本_i, 原型_j))
                for j in range(len(labels)):
                    # 样本与标签j的相似度: [num_samples_i]
                    cos_sim = (sample_emb @ label_emb[j:j+1].t()).squeeze(-1).detach().cpu().numpy()
                    # 取平均作为边权
                    sim_ll[i, j] = float(np.mean(cos_sim))
        else:
            # 如果没有训练样本，回退到简化版
            sim_ll = (label_emb @ label_emb.t()).detach().cpu().numpy()
    elif not use_field_level:
        # 直接计算标签间相似度
        sim_ll = (label_emb @ label_emb.t()).detach().cpu().numpy()
    
    np.fill_diagonal(sim_ll, -np.inf)

    # 每标签出边保留Top-p
    L = sim_ll.shape[0]
    k = max(1, int(math.ceil(L * float(edge_keep_ratio))))
    edges = []
    for i in range(L):
        idx = np.argsort(sim_ll[i])[::-1][:k]
        vals = sim_ll[i][idx]
        for j, v in zip(idx.tolist(), vals.tolist()):
            if not np.isfinite(v):
                continue
            edges.append({"src": int(i), "dst": int(j), "w": float(max(0.0, v))})

    return {
        "nodes": nodes,
        "edges": edges,
        "labels": labels,
        "label_nodes": {str(lbl): [i] for i, lbl in enumerate(labels)}
    }


