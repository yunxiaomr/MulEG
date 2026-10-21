import torch
from typing import List, Dict, Any


class CausalNode:
    def __init__(self, node_id: int, name: str, text: str,
                 text_embed: torch.Tensor = None):
        self.id = node_id
        self.name = name
        self.text = text
        self.text_embed = text_embed  # [d]


class CausalEdge:
    def __init__(self, src: int, dst: int, weight: float = 1.0):
        self.src = src
        self.dst = dst
        self.weight = float(weight)


class CausalDAG:
    def __init__(self, nodes: List[CausalNode], edges: List[CausalEdge],
                 labels: List[str] = None, label_nodes: Dict[str, List[int]] = None):
        self.nodes = nodes
        self.edges = edges
        self.num_nodes = len(nodes)
        self.id2idx = {n.id: i for i, n in enumerate(nodes)}
        self.adj = self._build_adj()
        self.labels = labels or []
        self.label_nodes = label_nodes or {}

    def _build_adj(self):
        adj = {i: [] for i in range(self.num_nodes)}
        for e in self.edges:
            adj[self.id2idx[e.src]].append(self.id2idx[e.dst])
        return adj

    def edge_index_and_weight(self):
        src_list = []
        dst_list = []
        w_list = []
        for e in self.edges:
            src_list.append(self.id2idx[e.src])
            dst_list.append(self.id2idx[e.dst])
            w_list.append(e.weight)
        if len(src_list) == 0:
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.tensor(w_list, dtype=torch.float)
        return edge_index, edge_weight

    def to_device(self, device: torch.device):
        for n in self.nodes:
            if n.text_embed is not None:
                n.text_embed = n.text_embed.to(device)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [{"id": n.id, "name": n.name, "text": n.text} for n in self.nodes],
            "edges": [{"src": e.src, "dst": e.dst, "w": e.weight} for e in self.edges],
            "labels": list(self.labels),
            "label_nodes": {k: list(v) for k, v in self.label_nodes.items()},
        }


