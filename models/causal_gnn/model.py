import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DirectedGATLayer


class CausalGNNModel(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, kappa: float = 1.0):
        super().__init__()
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.input_proj = nn.Linear(embed_dim + 1, hidden_dim)
        self.layers = nn.ModuleList([
            DirectedGATLayer(hidden_dim, hidden_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, dag, sample_scores):
        # dag: CausalDAG, sample_scores: [N] per node evidence s_v
        device = sample_scores.device
        text_emb = []
        for n in dag.nodes:
            if n.text_embed is None:
                raise ValueError("text_embed missing in DAG nodes")
            text_emb.append(n.text_embed)
        text_emb = torch.stack(text_emb, dim=0).to(device)
        h0 = torch.cat([text_emb, (self.kappa * sample_scores)[:, None]], dim=-1)
        h = self.input_proj(h0)

        edge_index, edge_weight = dag.edge_index_and_weight()
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device) if edge_weight.numel() > 0 else edge_weight

        attn_entropies = []
        for layer in self.layers:
            h, attn_in, attn_out = layer(h, edge_index, edge_weight)
            # 注意力熵（沿 edge 维聚合）：
            def entropy(a):
                a_clamped = a.clamp_min(1e-8)
                return (-a_clamped * a_clamped.log()).mean()
            attn_entropies.append(entropy(attn_in))
            attn_entropies.append(entropy(attn_out))

        # readout per node
        node_logits = self.readout(h).squeeze(-1)  # [N]
        # 返回节点logits与注意力熵列表用于正则
        return node_logits, attn_entropies


