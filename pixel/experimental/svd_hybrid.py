from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

from pixel.core.config import PIXELConfig
from pixel.core.patterns import PatternDictionary, PatternType
from pixel.core.synthesis import WeightSynthesizer
from pixel.utils.math_ops import svd_decompose, relative_error


class SVDHybridCompressor:
    def __init__(
        self,
        config: Optional[PIXELConfig] = None,
        svd_energy_threshold: float = 0.8,
        pattern_error_threshold: float = 0.05,
    ):
        self.config = config or PIXELConfig()
        self.device = self.config.resolved_device
        self.dtype = self.config.dtype
        self.svd_energy_threshold = svd_energy_threshold
        self.pattern_error_threshold = pattern_error_threshold

        self.pattern_dict = PatternDictionary(
            max_patterns=self.config.compression.max_patterns_per_layer * 2,
            device=self.device,
            dtype=self.dtype,
        )

        self.synthesizer = WeightSynthesizer(self.pattern_dict)

        for size in self.config.compression.pattern_sizes:
            self.pattern_dict.generate_base_patterns(size)

    def compress(
        self,
        weight: torch.Tensor,
        max_svd_rank: int = None,
    ) -> dict:
        weight = weight.to(device=self.device, dtype=self.dtype)
        original_shape = weight.shape

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)

        U, S, Vh = svd_decompose(weight, energy_threshold=self.svd_energy_threshold)
        svd_approx = U @ torch.diag(S) @ Vh

        if max_svd_rank and S.numel() > max_svd_rank:
            U = U[:, :max_svd_rank]
            S = S[:max_svd_rank]
            Vh = Vh[:max_svd_rank, :]
            svd_approx = U @ torch.diag(S) @ Vh

        residual = weight - svd_approx

        pattern_refs, final_residual = self.synthesizer.compress_weights(
            residual,
            max_patterns=self.config.compression.max_patterns_per_layer,
            error_tolerance=self.pattern_error_threshold,
        )

        reconstructed = svd_approx + self.synthesizer.synthesize(pattern_refs)
        if reconstructed is None:
            reconstructed = svd_approx

        error = relative_error(weight, reconstructed)

        original_size = weight.numel() * 4
        svd_size = (U.numel() + S.numel() + Vh.numel()) * 4
        pattern_size = len(pattern_refs) * 8
        residual_size = final_residual.numel() * 4 if torch.linalg.norm(final_residual) > 0.01 else 0
        compressed_size = svd_size + pattern_size + residual_size

        return {
            "svd": {"U": U, "S": S, "Vh": Vh},
            "pattern_refs": pattern_refs,
            "residual": final_residual if residual_size > 0 else None,
            "original_shape": original_shape,
            "svd_rank": S.numel(),
            "num_patterns": len(pattern_refs),
            "reconstruction_error": error,
            "compression_ratio": original_size / max(compressed_size, 1),
            "size_breakdown": {
                "original": original_size,
                "svd": svd_size,
                "patterns": pattern_size,
                "residual": residual_size,
                "total_compressed": compressed_size,
            },
        }

    def decompress(self, compressed: dict) -> torch.Tensor:
        svd = compressed["svd"]
        svd_approx = svd["U"] @ torch.diag(svd["S"]) @ svd["Vh"]

        pattern_part = self.synthesizer.synthesize(compressed["pattern_refs"])
        if pattern_part is None:
            pattern_part = torch.zeros_like(svd_approx)

        result = svd_approx + pattern_part

        if compressed.get("residual") is not None:
            result = result + compressed["residual"]

        original_shape = compressed.get("original_shape")
        if original_shape and result.shape != original_shape:
            result = result.view(original_shape)

        return result

    def compress_model_weights(
        self,
        weights: dict[str, torch.Tensor],
    ) -> dict[str, dict]:
        results = {}
        for name, weight in weights.items():
            results[name] = self.compress(weight)
        return results

    def decompress_model_weights(
        self,
        compressed_weights: dict[str, dict],
    ) -> dict[str, torch.Tensor]:
        results = {}
        for name, compressed in compressed_weights.items():
            results[name] = self.decompress(compressed)
        return results

    def stats(self) -> dict:
        return {
            "num_patterns": len(self.pattern_dict),
            "svd_energy_threshold": self.svd_energy_threshold,
            "pattern_error_threshold": self.pattern_error_threshold,
            "device": str(self.device),
        }
