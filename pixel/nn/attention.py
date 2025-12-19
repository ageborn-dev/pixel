from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatternAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)

        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class PatternSelector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        pattern_dim: int,
        num_patterns: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.pattern_embeddings = nn.Parameter(torch.randn(num_patterns, pattern_dim))
        
        self.attention = PatternAttention(hidden_dim, num_heads=4)
        
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        target_flat: torch.Tensor,
        top_k: int = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_flat.dim() == 1:
            target_flat = target_flat.unsqueeze(0)

        encoded = self.input_encoder(target_flat).unsqueeze(1)
        pattern_embed = self.pattern_embeddings.unsqueeze(0).expand(encoded.size(0), -1, -1)

        attended, weights = self.attention(encoded, pattern_embed, pattern_embed)
        
        scores = weights.mean(dim=1).squeeze(1)
        top_indices = torch.topk(scores, min(top_k, scores.size(-1)), dim=-1).indices

        selected_patterns = self.pattern_embeddings[top_indices.squeeze(0)]
        scales = self.scale_predictor(attended.squeeze(1)).squeeze(-1)

        return selected_patterns, scales
