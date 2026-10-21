"""
混合编码器 (Hybrid Encoder)
支持文本 (BERT) + 数值特征专门编码 + 符号图编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class NumericalFeatureEncoder(nn.Module):
    """
    数值特征专门编码器
    包含: 分箱编码 + 多项式特征 + 连续值编码
    支持视图消融: 可独立禁用任一视图，禁用视图输出零向量
    """

    def __init__(self, output_dim=768, num_bins=10,
                 use_binning=True, use_normalization=True,
                 use_polynomial=True, use_continuous=True, max_features=32):
        super().__init__()
        self.output_dim = output_dim
        self.num_bins = num_bins
        self.use_binning = use_binning
        self.use_normalization = use_normalization
        self.use_polynomial = use_polynomial
        self.use_continuous = use_continuous
        self.max_features = max_features

        # 1. 分箱编码（离散化）
        if use_binning:
            self.bin_embeddings = nn.ModuleList([
                nn.Embedding(num_bins, output_dim // 4)
                for _ in range(max_features)
            ])

        # 2. [FIXED] 归一化层: 移除BatchNorm1d(1)，因为它不支持动态特征数量
        # 我们将在 forward 中使用动态统计归一化

        # 3. 多项式特征: x, x^2, log(|x|+1), sqrt(|x|)
        if use_polynomial:
            self.poly_proj = nn.Linear(4, output_dim // 4)

        # 4. 连续值编码
        if use_continuous:
            self.continuous_encoder = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim // 4)
            )

        # 5. 特征融合 (fusion_input_dim 保持固定以兼容不同视图组合)
        # zero-out 禁用视图，fusion 层学习忽略零输入
        fusion_input_dim = output_dim // 4 * 3  # bin + poly + continuous 各占 output_dim//4
        self.per_view_dim = output_dim // 4

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 记录实际可用视图
        self._active_views = {
            'binning': use_binning,
            'polynomial': use_polynomial,
            'continuous': use_continuous
        }

    def set_view_mode(self, mode: str):
        """动态设置视图模式用于消融实验.

        Args:
            mode: 'full' | 'no_bin' | 'no_high_order' | 'no_continuous' |
                  'bin_only' | 'high_order_only' | 'continuous_only'
        """
        if mode == 'full':
            self._active_views = {'binning': self.use_binning,
                                  'polynomial': self.use_polynomial,
                                  'continuous': self.use_continuous}
        elif mode == 'no_bin':
            self._active_views = {'binning': False,
                                  'polynomial': self.use_polynomial,
                                  'continuous': self.use_continuous}
        elif mode == 'no_high_order':
            self._active_views = {'binning': self.use_binning,
                                  'polynomial': False,
                                  'continuous': self.use_continuous}
        elif mode == 'no_continuous':
            self._active_views = {'binning': self.use_binning,
                                  'polynomial': self.use_polynomial,
                                  'continuous': False}
        elif mode == 'bin_only':
            self._active_views = {'binning': self.use_binning,
                                  'polynomial': False,
                                  'continuous': False}
        elif mode == 'high_order_only':
            self._active_views = {'binning': False,
                                  'polynomial': self.use_polynomial,
                                  'continuous': False}
        elif mode == 'continuous_only':
            self._active_views = {'binning': False,
                                  'polynomial': False,
                                  'continuous': self.use_continuous}
        else:
            raise ValueError(f"Unknown view_mode: {mode}")
    
    def forward(self, x, feature_mask=None):
        """
        Args:
            x: (batch_size, num_features) 数值特征
            feature_mask: (num_features,) bool
        """
        B, F = x.shape
        device = x.device

        if feature_mask is not None:
            # [FIXED] 维度修复: 处理 2D mask
            if feature_mask.dim() > 1:
                mask_1d = feature_mask[0].bool()
            else:
                mask_1d = feature_mask.bool()

            # 使用 1D mask 选择数值列 -> (Batch, Num_Numeric_Features)
            numeric_x = x[:, mask_1d]
        else:
            numeric_x = x

        if numeric_x.size(1) == 0:
            return torch.zeros(B, self.output_dim, device=device)

        # 归一化 (供 polynomial 和 continuous 使用)
        if self.use_normalization and B > 1:
            mean = numeric_x.mean(dim=0, keepdim=False)
            std = numeric_x.std(dim=0, keepdim=False) + 1e-5
            numeric_x_norm = (numeric_x - mean) / std
        else:
            numeric_x_norm = numeric_x

        # 1. 分箱编码 (discretized-bin view)
        if self._active_views.get('binning', False) and hasattr(self, 'bin_embeddings'):
            min_val = numeric_x.min(dim=0, keepdim=True)[0]
            max_val = numeric_x.max(dim=0, keepdim=True)[0]
            range_val = max_val - min_val + 1e-8
            normalized = (numeric_x - min_val) / range_val
            normalized = normalized.clamp(0, 1)

            bin_indices = (normalized * (self.num_bins - 1)).long()
            bin_indices = bin_indices.clamp(0, self.num_bins - 1)

            bin_embs = []
            limit = min(numeric_x.size(1), len(self.bin_embeddings))
            for i in range(limit):
                bin_embs.append(self.bin_embeddings[i](bin_indices[:, i]))

            if len(bin_embs) > 0:
                bin_emb = torch.stack(bin_embs, dim=1).mean(dim=1)  # [B, D//4]
            else:
                bin_emb = torch.zeros(B, self.per_view_dim, device=device)
        else:
            bin_emb = torch.zeros(B, self.per_view_dim, device=device)

        # 2. 多项式特征 (high-order view)
        if self._active_views.get('polynomial', False) and hasattr(self, 'poly_proj'):
            poly_features = torch.stack([
                numeric_x_norm,
                numeric_x_norm ** 2,
                torch.log(torch.abs(numeric_x_norm) + 1),
                torch.sqrt(torch.abs(numeric_x_norm) + 1e-8)
            ], dim=-1)
            poly_emb = self.poly_proj(poly_features).mean(dim=1)  # [B, D//4]
        else:
            poly_emb = torch.zeros(B, self.per_view_dim, device=device)

        # 3. 连续值编码 (continuous view)
        if self._active_views.get('continuous', False) and hasattr(self, 'continuous_encoder'):
            continuous_emb = self.continuous_encoder(
                numeric_x_norm.unsqueeze(-1)
            ).mean(dim=1)  # [B, D//4]
        else:
            continuous_emb = torch.zeros(B, self.per_view_dim, device=device)

        # 4. 融合 (concatenate in fixed order: bin, poly, continuous)
        fused = torch.cat([bin_emb, poly_emb, continuous_emb], dim=-1)
        output = self.fusion(fused)

        return output


class HybridEncoder(nn.Module):
    def __init__(self, bert_encoder, embed_dim=768,
                 use_cross_attention=True, cross_attn_heads=4,
                 view_mode='full'):
        super().__init__()
        self.bert = bert_encoder
        self.embed_dim = embed_dim
        self.use_cross_attention = use_cross_attention

        self.numeric_encoder = NumericalFeatureEncoder(
            output_dim=embed_dim,
            use_binning=True,
            use_normalization=True,
            use_polynomial=True,
            use_continuous=True
        )
        self.numeric_encoder.set_view_mode(view_mode)

        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=cross_attn_heads,
                dropout=0.1,
                batch_first=True
            )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def set_view_mode(self, mode: str):
        """设置数值特征编码的视图模式 (用于消融实验)."""
        self.numeric_encoder.set_view_mode(mode)
    
    def forward(self, text_input, numeric_features=None, feature_mask=None):
        bert_outputs = self.bert(**text_input)
        text_emb = bert_outputs.pooler_output
        
        if numeric_features is not None and numeric_features.size(1) > 0:
            numeric_emb = self.numeric_encoder(numeric_features, feature_mask)
            
            if self.use_cross_attention:
                q = text_emb.unsqueeze(1)
                k = numeric_emb.unsqueeze(1)
                v = numeric_emb.unsqueeze(1)
                attn_output, _ = self.cross_attention(query=q, key=k, value=v)
                attn_output = attn_output.squeeze(1)
            else:
                attn_output = numeric_emb
            
            combined = torch.cat([text_emb, attn_output], dim=-1)
            
            if combined.size(0) > 1:
                fused_emb = self.fusion(combined)
            else:
                fused_emb = F.relu(self.fusion[0](combined))
                fused_emb = self.fusion[3](fused_emb)
        else:
            fused_emb = text_emb
        
        return fused_emb


def identify_numeric_features(sample_dict: Dict, task_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    masks = []
    for key, value in sample_dict.items():
        if key in ['lbl', 'target_key', 'label']:
            continue
        try:
            if isinstance(value, (int, float)):
                features.append(float(value))
                masks.append(True)
            elif isinstance(value, str):
                numeric_value = float(value)
                features.append(numeric_value)
                masks.append(True)
        except (ValueError, TypeError):
            features.append(0.0)
            masks.append(False)
    
    if len(features) == 0:
        return torch.zeros(1), torch.zeros(1, dtype=torch.bool)
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(masks, dtype=torch.bool)