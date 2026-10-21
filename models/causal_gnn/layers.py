import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectedGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4,
                 attn_bias_scale: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.q_in = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.k_in = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.v_in = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        self.q_out = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.k_out = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.v_out = nn.Linear(in_dim, out_dim * num_heads, bias=False)

        self.proj = nn.Linear(out_dim * num_heads * 2, out_dim)
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_bias_scale = attn_bias_scale

    def _attend(self, h_src, h_dst, edge_index, edge_weight, q_layer, k_layer, v_layer):
        # h_src/h_dst: [N, D]
        N = h_src.size(0)
        H = self.num_heads
        D = q_layer.out_features // H

        Q = q_layer(h_dst).view(N, H, D)
        K = k_layer(h_src).view(N, H, D)
        V = v_layer(h_src).view(N, H, D)

        src, dst = edge_index  # [E]
        q = Q[dst]  # [E, H, D]
        k = K[src]  # [E, H, D]
        v = V[src]  # [E, H, D]

        attn_logits = (q * k).sum(-1) / (D ** 0.5)
        if edge_weight is not None and edge_weight.numel() == attn_logits.size(0):
            # 使用 log-bias 提升数值稳定性，并放大/压缩差异
            ew = edge_weight.to(h_src.device)
            ew_bias = torch.log(ew.clamp_min(1e-8))  # [E]
            attn_logits = attn_logits + self.attn_bias_scale * ew_bias[:, None]

        attn = F.softmax(attn_logits, dim=0)  # normalized over edges list
        m = attn[..., None] * v  # [E, H, D]

        out = torch.zeros(N, H, D, device=h_src.device)
        out = out.index_add(0, dst, m)  # sum over edges
        out = out.reshape(N, H * D)
        return out, attn

    def forward(self, h, edge_index, edge_weight):
        # in-channel: parents->child
        m_in, attn_in = self._attend(h, h, edge_index, edge_weight, self.q_in, self.k_in, self.v_in)
        # out-channel: child->parents (use reversed edges)
        if edge_index.numel() == 0:
            edge_index_rev = edge_index
            edge_weight_rev = edge_weight
        else:
            edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
            edge_weight_rev = edge_weight
        m_out, attn_out = self._attend(h, h, edge_index_rev, edge_weight_rev, self.q_out, self.k_out, self.v_out)

        h_new = torch.cat([m_in, m_out], dim=-1)
        h_new = self.proj(h_new)
        h_new = self.dropout(h_new)
        h_new = self.ln(h + h_new)
        return h_new, attn_in, attn_out


