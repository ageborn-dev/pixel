from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from pixel.core.patterns import PatternDictionary
from pixel.core.synthesis import WeightSynthesizer


class PIXELLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        pattern_dict: PatternDictionary,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pattern_dict = pattern_dict
        self.synthesizer = WeightSynthesizer(pattern_dict)

        self.pattern_refs: list[tuple[int, float]] = []
        self.residual: Optional[torch.Tensor] = None
        self._cached_weight: Optional[torch.Tensor] = None
        self._cache_valid = False

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_identity()

    def _init_identity(self) -> None:
        size = min(self.in_features, self.out_features)
        identity = torch.eye(size, device=self.pattern_dict.device)
        pid = self.pattern_dict.add(identity)
        self.pattern_refs = [(pid, 1.0)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.get_weight()
        out = F.linear(x, weight, self.bias)
        return out

    def get_weight(self) -> torch.Tensor:
        if self._cache_valid and self._cached_weight is not None:
            return self._cached_weight

        weight = self.synthesizer.synthesize(self.pattern_refs)
        if weight is None:
            weight = torch.zeros(
                self.out_features, self.in_features,
                device=self.pattern_dict.device,
            )

        if self.residual is not None:
            weight = weight + self.residual

        target_shape = (self.out_features, self.in_features)
        if weight.shape != target_shape:
            weight = self._reshape_weight(weight, target_shape)

        self._cached_weight = weight
        self._cache_valid = True
        return weight

    def _reshape_weight(
        self,
        weight: torch.Tensor,
        target_shape: tuple[int, int],
    ) -> torch.Tensor:
        current = weight.flatten()
        target_size = target_shape[0] * target_shape[1]

        if current.numel() >= target_size:
            return current[:target_size].view(target_shape)

        padded = F.pad(current, (0, target_size - current.numel()))
        return padded.view(target_shape)

    def compress_from_linear(self, linear: nn.Linear, max_patterns: int = 8) -> float:
        weight = linear.weight.data.to(self.pattern_dict.device)
        refs, residual = self.synthesizer.compress_weights(weight, max_patterns)

        self.pattern_refs = refs
        residual_norm = torch.linalg.norm(residual)
        weight_norm = torch.linalg.norm(weight)

        if residual_norm / weight_norm > 0.01:
            self.residual = nn.Parameter(residual, requires_grad=False)
        else:
            self.residual = None

        if linear.bias is not None and self.bias is not None:
            self.bias.data = linear.bias.data.clone()

        self._cache_valid = False
        return (residual_norm / weight_norm).item()

    def invalidate_cache(self) -> None:
        self._cache_valid = False
        self._cached_weight = None


class PIXELLinear(PIXELLayer):
    pass


def convert_linear_to_pixel(
    linear: nn.Linear,
    pattern_dict: PatternDictionary,
    max_patterns: int = 8,
) -> tuple[PIXELLinear, float]:
    pixel_layer = PIXELLinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        pattern_dict=pattern_dict,
        bias=linear.bias is not None,
    )
    error = pixel_layer.compress_from_linear(linear, max_patterns)
    return pixel_layer, error


def replace_linear_layers(
    module: nn.Module,
    pattern_dict: PatternDictionary,
    max_patterns: int = 8,
    prefix: str = "",
) -> dict[str, float]:
    errors = {}

    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            pixel_layer, error = convert_linear_to_pixel(child, pattern_dict, max_patterns)
            setattr(module, name, pixel_layer)
            errors[full_name] = error
        else:
            child_errors = replace_linear_layers(child, pattern_dict, max_patterns, full_name)
            errors.update(child_errors)

    return errors
