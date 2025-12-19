from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum
import torch


class DeviceType(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class CompressionConfig:
    target_ratio: float = 0.1
    quality_threshold: float = 0.95
    max_patterns_per_layer: int = 16
    pattern_sizes: tuple[int, ...] = (4, 8, 16)
    use_svd_init: bool = True
    svd_rank_ratio: float = 0.1
    pruning_threshold: int = 5
    cache_size: int = 1024

    def validate(self) -> None:
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if not 0 < self.quality_threshold <= 1:
            raise ValueError("quality_threshold must be in (0, 1]")
        if self.max_patterns_per_layer < 1:
            raise ValueError("max_patterns_per_layer must be >= 1")


@dataclass
class QuantizationConfig:
    enabled: bool = False
    pattern_bits: int = 8
    scale_bits: int = 16
    dynamic_range: bool = True


@dataclass
class PIXELConfig:
    matrix_size: int = 256
    hidden_size: int = 768
    num_layers: int = 12
    pattern_embed_dim: int = 128
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    device: DeviceType = DeviceType.AUTO
    dtype: torch.dtype = torch.float32
    seed: int | None = None

    def __post_init__(self) -> None:
        self.compression.validate()
        if self.seed is not None:
            torch.manual_seed(self.seed)

    @property
    def resolved_device(self) -> torch.device:
        if self.device == DeviceType.AUTO:
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.device.value)

    def to_dict(self) -> dict:
        return {
            "matrix_size": self.matrix_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "pattern_embed_dim": self.pattern_embed_dim,
            "compression": {
                "target_ratio": self.compression.target_ratio,
                "quality_threshold": self.compression.quality_threshold,
                "max_patterns_per_layer": self.compression.max_patterns_per_layer,
                "pattern_sizes": list(self.compression.pattern_sizes),
                "use_svd_init": self.compression.use_svd_init,
                "svd_rank_ratio": self.compression.svd_rank_ratio,
            },
            "quantization": {
                "enabled": self.quantization.enabled,
                "pattern_bits": self.quantization.pattern_bits,
                "scale_bits": self.quantization.scale_bits,
            },
            "device": self.device.value,
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PIXELConfig:
        compression = CompressionConfig(**data.get("compression", {}))
        quantization = QuantizationConfig(**data.get("quantization", {}))
        return cls(
            matrix_size=data.get("matrix_size", 256),
            hidden_size=data.get("hidden_size", 768),
            num_layers=data.get("num_layers", 12),
            pattern_embed_dim=data.get("pattern_embed_dim", 128),
            compression=compression,
            quantization=quantization,
            device=DeviceType(data.get("device", "auto")),
        )

    @classmethod
    def for_small_model(cls) -> PIXELConfig:
        return cls(
            matrix_size=64,
            hidden_size=256,
            num_layers=4,
            compression=CompressionConfig(
                target_ratio=0.15,
                max_patterns_per_layer=8,
                pattern_sizes=(2, 4, 8),
            ),
        )

    @classmethod
    def for_large_model(cls) -> PIXELConfig:
        return cls(
            matrix_size=512,
            hidden_size=4096,
            num_layers=32,
            compression=CompressionConfig(
                target_ratio=0.05,
                max_patterns_per_layer=32,
                pattern_sizes=(8, 16, 32),
                use_svd_init=True,
            ),
        )
