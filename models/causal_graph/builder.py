import re
import json
from typing import Dict, List, Tuple, Callable
import torch
from transformers import AutoTokenizer, AutoModel

from .schemas import CausalNode, CausalEdge, CausalDAG


def simple_text_embed(tokenizer, encoder, texts: List[str], device) -> torch.Tensor:
    with torch.no_grad():
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = encoder(**inputs)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        return outputs.last_hidden_state[:, 0]


def extract_variables_from_explanations(explanations: List[str]) -> List[str]:
    vars_set = set()
    for s in explanations:
        s_norm = s.strip()
        if len(s_norm) == 0:
            continue
        vars_set.add(s_norm)
    return sorted(list(vars_set))


# 先验模式枚举（名称 -> 生成函数）
StructurePrior = Callable[[List[str]], List[Tuple[int, int, float]]]


def prior_chain(variables: List[str]) -> List[Tuple[int, int, float]]:
    n = len(variables)
    edges = []
    for i in range(n - 1):
        edges.append((i, i + 1, 1.0))
    return edges


def prior_skip_chain(variables: List[str]) -> List[Tuple[int, int, float]]:
    n = len(variables)
    edges = prior_chain(variables)
    for i in range(n - 2):
        edges.append((i, i + 2, 0.6))
    return edges


def prior_fork(variables: List[str]) -> List[Tuple[int, int, float]]:
    # 0 -> others
    n = len(variables)
    edges = []
    for j in range(1, n):
        edges.append((0, j, 0.8))
    return edges


def prior_merge(variables: List[str]) -> List[Tuple[int, int, float]]:
    # others -> last
    n = len(variables)
    edges = []
    for i in range(n - 1):
        edges.append((i, n - 1, 0.8))
    return edges


def prior_v_structure(variables: List[str]) -> List[Tuple[int, int, float]]:
    # 0->2 <-1 (若存在)
    edges = []
    if len(variables) >= 3:
        edges.append((0, 2, 0.9))
        edges.append((1, 2, 0.9))
    return edges


def prior_two_layer_dag(variables: List[str]) -> List[Tuple[int, int, float]]:
    # 前半层 -> 后半层
    n = len(variables)
    mid = n // 2
    edges = []
    for i in range(mid):
        for j in range(mid, n):
            edges.append((i, j, 0.5))
    return edges


def prior_chain_with_backoff(variables: List[str]) -> List[Tuple[int, int, float]]:
    # 链 + 小概率更远跳边
    n = len(variables)
    edges = prior_chain(variables)
    for i in range(n - 3):
        edges.append((i, i + 3, 0.4))
    return edges


def prior_local_window(variables: List[str], win: int = 3) -> List[Tuple[int, int, float]]:
    n = len(variables)
    edges = []
    for i in range(n):
        for j in range(i + 1, min(n, i + 1 + win)):
            edges.append((i, j, 0.6))
    return edges


def prior_star_tail(variables: List[str]) -> List[Tuple[int, int, float]]:
    # 0 -> 中间若干 -> n-1
    n = len(variables)
    edges = []
    for j in range(1, n - 1):
        edges.append((0, j, 0.6))
        edges.append((j, n - 1, 0.6))
    return edges


def prior_random_sparse(variables: List[str]) -> List[Tuple[int, int, float]]:
    import random
    n = len(variables)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.15:
                edges.append((i, j, 0.5))
    return edges


PRIOR_REGISTRY: Dict[str, StructurePrior] = {
    "chain": prior_chain,
    "skip_chain": prior_skip_chain,
    "fork": prior_fork,
    "merge": prior_merge,
    "v_structure": prior_v_structure,
    "two_layer": prior_two_layer_dag,
    "chain_backoff": prior_chain_with_backoff,
    "local_window": prior_local_window,
    "star_tail": prior_star_tail,
    "random_sparse": prior_random_sparse,
}


def propose_edges_by_templates(variables: List[str], prior: str = "skip_chain") -> List[Tuple[int, int, float]]:
    # 兼容原实现：默认使用 skip_chain；支持 10+ 先验
    if prior in PRIOR_REGISTRY:
        edges = PRIOR_REGISTRY[prior](variables)
    else:
        edges = prior_skip_chain(variables)
    # 只保留 i<j 的边以避免明显环
    edges = [(i, j, w) for (i, j, w) in edges if i < j]
    return edges


# 语义方向与边权估计（规则骨架占位）：
SEM_DIR_PATTERNS = [
    (re.compile(r"(.+?)\s*(causes|leads to|results in|induces)\s*(.+)", re.I), +1),
    (re.compile(r"(.+?)\s*(because|due to)\s*(.+)", re.I), -1),  # v because u 近似 u->v
]


def estimate_edge_semantics(u_text: str, v_text: str) -> Tuple[int, float]:
    # 返回 (direction_sign, weight_bias)
    text = f"{u_text} {v_text}"
    for pat, sign in SEM_DIR_PATTERNS:
        if pat.search(text):
            return sign, 0.2
    return +1, 0.0


def apply_semantic_bias(variables: List[str], edges: List[Tuple[int, int, float]]):
    adjusted = []
    for (i, j, w) in edges:
        sign, bias = estimate_edge_semantics(variables[i], variables[j])
        w_new = max(0.0, w + bias)
        adjusted.append((i, j, w_new))
    return adjusted


# 样本证据一致性评估骨架（占位，无样本时跳过）
def evidence_consistency_prune(edges: List[Tuple[int, int, float]], contribution_scores: Dict[Tuple[int, int], float] = None,
                               prune_threshold: float = 0.0):
    if not contribution_scores:
        return edges
    kept = []
    for (i, j, w) in edges:
        contrib = contribution_scores.get((i, j), 0.0)
        if contrib >= prune_threshold:
            kept.append((i, j, w))
    return kept


# 稀疏化（L1 先验近似）：按权重从小到大裁剪到按密度保留
def l1_sparsify(edges: List[Tuple[int, int, float]], keep_ratio: float = 0.6):
    if not edges:
        return edges
    k = max(1, int(len(edges) * keep_ratio))
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
    return edges_sorted[:k]


def break_cycles(edges: List[Tuple[int, int, float]], num_nodes: int) -> List[Tuple[int, int, float]]:
    # 拓扑排序检测环，若有环则移除最小权边
    adj = {i: [] for i in range(num_nodes)}
    indeg = [0] * num_nodes
    for u, v, w in edges:
        adj[u].append((v, w))
        indeg[v] += 1
    order = []
    q = [i for i in range(num_nodes) if indeg[i] == 0]
    while q:
        u = q.pop(0)
        order.append(u)
        for v, w in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) == num_nodes:
        return edges
    # 存在环：移除最小权值边并递归
    edges_sorted = sorted(edges, key=lambda x: x[2], reverse=True)
    edges_sorted.pop()  # 移除一条最小权边
    return break_cycles(edges_sorted, num_nodes)


def build_task_level_dag(task_name: str,
                         guidemap: Dict[str, str],
                         tokenizer, encoder, device,
                         prior: str = "skip_chain",
                         enable_semantic: bool = True,
                         keep_ratio: float = 0.8) -> CausalDAG:
    # 1) 变量集合
    all_exps = list(guidemap.keys())
    variables = extract_variables_from_explanations(all_exps)

    # 2) 初步边（结构先验）
    raw_edges = propose_edges_by_templates(variables, prior=prior)

    # 2.1 语义方向/权重 bias（占位规则）
    if enable_semantic:
        raw_edges = apply_semantic_bias(variables, raw_edges)

    # 2.2 稀疏化（近似 L1）
    raw_edges = l1_sparsify(raw_edges, keep_ratio=keep_ratio)

    # 2.3 无环修正
    raw_edges = break_cycles(raw_edges, len(variables))

    # 3) 文本嵌入
    text_emb = simple_text_embed(tokenizer, encoder, variables, device)

    # 4) 组装 DAG
    nodes = []
    for i, txt in enumerate(variables):
        nodes.append(CausalNode(node_id=i, name=f"var_{i}", text=txt, text_embed=text_emb[i]))
    edges = []
    for src, dst, w in raw_edges:
        edges.append(CausalEdge(src=src, dst=dst, weight=w))
    dag = CausalDAG(nodes, edges)
    return dag


