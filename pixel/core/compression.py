from __future__ import annotations
from typing import Optional, Callable
from dataclasses import dataclass
import torch

from pixel.core.config import PIXELConfig
from pixel.core.patterns import PatternDictionary, PatternType
from pixel.core.synthesis import WeightSynthesizer


@dataclass
class CompressionResult:
    original_size: int
    compressed_size: int
    num_patterns: int
    reconstruction_error: float
    compression_ratio: float
    pattern_refs: dict[str, list[tuple[int, float]]]

    def summary(self) -> str:
        return (
            f"Compression: {self.compression_ratio:.2f}x | "
            f"Error: {self.reconstruction_error:.4f} | "
            f"Patterns: {self.num_patterns}"
        )


class PIXELCompressor:
    def __init__(self, config: Optional[PIXELConfig] = None):
        self.config = config or PIXELConfig()
        self.device = self.config.resolved_device
        self.dtype = self.config.dtype

        self.pattern_dict = PatternDictionary(
            max_patterns=self.config.compression.max_patterns_per_layer * self.config.num_layers,
            cache_size=self.config.compression.cache_size,
            device=self.device,
            dtype=self.dtype,
        )

        self.synthesizer = WeightSynthesizer(
            pattern_dict=self.pattern_dict,
            cache_size=self.config.compression.cache_size,
        )

        self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        for size in self.config.compression.pattern_sizes:
            self.pattern_dict.generate_base_patterns(size)

    def compress(
        self,
        weights: dict[str, torch.Tensor],
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> CompressionResult:
        pattern_refs = {}
        total_original = 0
        total_compressed = 0
        total_error = 0.0
        num_layers = len(weights)

        for idx, (name, weight) in enumerate(weights.items()):
            weight = weight.to(device=self.device, dtype=self.dtype)
            original_numel = weight.numel()
            total_original += original_numel * 4

            refs, residual = self.synthesizer.compress_weights(
                weight,
                max_patterns=self.config.compression.max_patterns_per_layer,
                error_tolerance=1 - self.config.compression.quality_threshold,
            )

            pattern_refs[name] = refs

            compressed_size = len(refs) * 8  # id + scale
            if torch.linalg.norm(residual) / torch.linalg.norm(weight) > 0.01:
                compressed_size += residual.numel() * 4
            total_compressed += compressed_size

            error = torch.linalg.norm(residual) / (torch.linalg.norm(weight) + 1e-8)
            total_error += error.item()

            if progress_callback:
                progress_callback(name, (idx + 1) / num_layers)

        avg_error = total_error / max(num_layers, 1)
        compression_ratio = total_original / max(total_compressed, 1)

        return CompressionResult(
            original_size=total_original,
            compressed_size=total_compressed,
            num_patterns=len(self.pattern_dict),
            reconstruction_error=avg_error,
            compression_ratio=compression_ratio,
            pattern_refs=pattern_refs,
        )

    def decompress(
        self,
        pattern_refs: dict[str, list[tuple[int, float]]],
        residuals: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        weights = {}
        residuals = residuals or {}

        for name, refs in pattern_refs.items():
            synthesized = self.synthesizer.synthesize(refs)
            if synthesized is None:
                continue

            if name in residuals:
                synthesized = synthesized + residuals[name]

            weights[name] = synthesized

        return weights

    def compress_tensor(
        self,
        tensor: torch.Tensor,
        max_patterns: int = None,
    ) -> tuple[list[tuple[int, float]], torch.Tensor, float]:
        tensor = tensor.to(device=self.device, dtype=self.dtype)
        max_patterns = max_patterns or self.config.compression.max_patterns_per_layer

        refs, residual = self.synthesizer.compress_weights(
            tensor,
            max_patterns=max_patterns,
            error_tolerance=1 - self.config.compression.quality_threshold,
        )

        error = torch.linalg.norm(residual) / (torch.linalg.norm(tensor) + 1e-8)
        return refs, residual, error.item()

    def decompress_tensor(
        self,
        refs: list[tuple[int, float]],
        residual: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        result = self.synthesizer.synthesize(refs)
        if result is not None and residual is not None:
            result = result + residual
        return result

    def save(self, path: str) -> None:
        state = {
            "config": self.config.to_dict(),
            "pattern_dict": self.pattern_dict.state_dict(),
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> PIXELCompressor:
        state = torch.load(path, weights_only=False)
        config = PIXELConfig.from_dict(state["config"])
        compressor = cls(config)
        compressor.pattern_dict.load_state_dict(state["pattern_dict"])
        return compressor

    def stats(self) -> dict:
        return {
            "num_patterns": len(self.pattern_dict),
            "device": str(self.device),
            "config": self.config.to_dict(),
        }
