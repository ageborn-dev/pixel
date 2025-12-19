from __future__ import annotations
from typing import Optional
from collections import OrderedDict
import torch
import torch.nn as nn

from pixel.core.patterns import PatternDictionary


class WeightSynthesizer:
    def __init__(
        self,
        pattern_dict: PatternDictionary,
        cache_size: int = 512,
        optimization_steps: int = 10,
    ):
        self.pattern_dict = pattern_dict
        self.cache_size = cache_size
        self.optimization_steps = optimization_steps
        self._cache: OrderedDict[tuple, torch.Tensor] = OrderedDict()

    def synthesize(
        self,
        pattern_refs: list[tuple[int, float]],
        use_cache: bool = True,
    ) -> Optional[torch.Tensor]:
        if not pattern_refs:
            return None

        cache_key = tuple(sorted(pattern_refs))
        if use_cache and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        result = None
        for pattern_id, scale in pattern_refs:
            pattern = self.pattern_dict.get(pattern_id)
            if pattern is None:
                continue
            contribution = pattern * scale
            if result is None:
                result = contribution
            else:
                result = result + contribution

        if result is not None and use_cache:
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)
            self._cache[cache_key] = result

        return result

    def optimize_scales(
        self,
        target: torch.Tensor,
        pattern_ids: list[int],
        lr: float = 0.1,
    ) -> list[tuple[int, float]]:
        patterns = []
        valid_ids = []

        for pid in pattern_ids:
            pattern = self.pattern_dict.get(pid)
            if pattern is not None and pattern.shape == target.shape:
                patterns.append(pattern.flatten())
                valid_ids.append(pid)

        if not patterns:
            return []

        P = torch.stack(patterns, dim=1)
        target_flat = target.flatten()

        try:
            result = torch.linalg.lstsq(P, target_flat.unsqueeze(1))
            scales = result.solution.squeeze(1)
        except RuntimeError:
            PtP = P.T @ P
            regularization = 1e-6 * torch.eye(PtP.shape[0], device=P.device, dtype=P.dtype)
            scales = torch.linalg.solve(PtP + regularization, P.T @ target_flat)

        return [(pid, scale.item()) for pid, scale in zip(valid_ids, scales)]

    def compress_weights(
        self,
        weights: torch.Tensor,
        max_patterns: int = 8,
        error_tolerance: float = 0.05,
    ) -> tuple[list[tuple[int, float]], torch.Tensor]:
        residual = weights.clone()
        selected_refs = []

        matching_patterns = [
            pid for pid, p in self.pattern_dict.patterns.items()
            if p.shape == weights.shape
        ]

        if not matching_patterns:
            self._generate_patterns_for_shape(weights.shape)

        for _ in range(max_patterns):
            residual_norm = torch.linalg.norm(residual)
            original_norm = torch.linalg.norm(weights)

            if residual_norm / original_norm < error_tolerance:
                break

            matches = self.pattern_dict.find_best_matches(residual, top_k=1, threshold=0.01)
            if not matches:
                u, s, vh = torch.linalg.svd(residual, full_matrices=False)
                if s.numel() > 0:
                    rank1_pattern = torch.outer(u[:, 0], vh[0, :]) * s[0]
                    pid = self.pattern_dict.add(rank1_pattern)
                    matches = [(pid, 1.0)]
                else:
                    break

            pattern_id, similarity = matches[0]
            pattern = self.pattern_dict.get(pattern_id)

            if pattern is None:
                break

            if pattern.shape != residual.shape:
                continue

            pattern_flat = pattern.flatten()
            residual_flat = residual.flatten()
            scale = torch.dot(residual_flat, pattern_flat) / torch.dot(pattern_flat, pattern_flat)
            scale = scale.item()

            selected_refs.append((pattern_id, scale))
            residual = residual - pattern * scale

        if len(selected_refs) > 1:
            pattern_ids = [ref[0] for ref in selected_refs]
            optimized = self.optimize_scales(weights, pattern_ids)
            if optimized:
                selected_refs = optimized
                synthesized = self.synthesize(selected_refs, use_cache=False)
                if synthesized is not None:
                    residual = weights - synthesized

        return selected_refs, residual

    def _generate_patterns_for_shape(self, shape: tuple[int, ...]) -> None:
        if len(shape) != 2:
            return
        rows, cols = shape
        device = self.pattern_dict.device
        dtype = self.pattern_dict.dtype

        size = min(rows, cols)
        identity = torch.eye(size, device=device, dtype=dtype)
        if rows != cols:
            padded = torch.zeros(rows, cols, device=device, dtype=dtype)
            padded[:size, :size] = identity
            identity = padded
        self.pattern_dict.add(identity)

        x = torch.linspace(0, 2 * 3.14159, cols, device=device, dtype=dtype)
        y = torch.linspace(0, 2 * 3.14159, rows, device=device, dtype=dtype)
        for freq in [1, 2]:
            sin_pattern = torch.outer(torch.sin(freq * y), torch.ones_like(x))
            cos_pattern = torch.outer(torch.cos(freq * y), torch.ones_like(x))
            self.pattern_dict.add(sin_pattern)
            self.pattern_dict.add(cos_pattern)

    def clear_cache(self) -> None:
        self._cache.clear()


class NeuralSynthesizer(nn.Module):
    def __init__(
        self,
        pattern_dict: PatternDictionary,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        self.pattern_dict = pattern_dict
        self.embed_dim = embed_dim

        max_patterns = pattern_dict.max_patterns
        self.pattern_embeddings = nn.Embedding(max_patterns, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.synthesizer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.scale_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        target_embedding: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pattern_embeds = self.pattern_embeddings(candidate_ids)
        
        target_expanded = target_embedding.unsqueeze(1)
        attended, attention_weights = self.attention(
            target_expanded, pattern_embeds, pattern_embeds
        )
        
        combined = self.synthesizer(attended.squeeze(1))
        
        target_repeated = target_embedding.unsqueeze(1).expand_as(pattern_embeds)
        pair_features = torch.cat([target_repeated, pattern_embeds], dim=-1)
        scales = self.scale_predictor(pair_features).squeeze(-1)

        return combined, scales

    def encode_target(self, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 2:
            target = target.flatten()
        embedding_size = min(target.numel(), self.embed_dim)
        return target[:embedding_size].unsqueeze(0)
