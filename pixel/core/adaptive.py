from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


class AdaptiveCompressor:
    def __init__(
        self,
        error_threshold: float = 0.05,
        min_compression_ratio: float = 2.0,
        quantize_patterns: bool = True,
        pattern_bits: int = 8,
        layer_configs: dict = None,
    ):
        self.error_threshold = error_threshold
        self.min_compression_ratio = min_compression_ratio
        self.quantize_patterns = quantize_patterns
        self.pattern_bits = pattern_bits
        
        self.layer_configs = layer_configs or {
            "attn": {"svd_energy": 0.90, "max_rank_ratio": 0.4},
            "mlp": {"svd_energy": 0.92, "max_rank_ratio": 0.5},
            "embed": {"svd_energy": 0.95, "max_rank_ratio": 0.7},
            "default": {"svd_energy": 0.90, "max_rank_ratio": 0.5},
        }

    @classmethod
    def high_quality(cls) -> "AdaptiveCompressor":
        return cls(
            error_threshold=0.01,
            layer_configs={
                "attn": {"svd_energy": 0.98, "max_rank_ratio": 0.8},
                "mlp": {"svd_energy": 0.99, "max_rank_ratio": 0.85},
                "embed": {"svd_energy": 0.995, "max_rank_ratio": 0.9},
                "default": {"svd_energy": 0.98, "max_rank_ratio": 0.8},
            },
        )

    @classmethod
    def balanced(cls) -> "AdaptiveCompressor":
        return cls(
            error_threshold=0.03,
            layer_configs={
                "attn": {"svd_energy": 0.92, "max_rank_ratio": 0.5},
                "mlp": {"svd_energy": 0.94, "max_rank_ratio": 0.6},
                "embed": {"svd_energy": 0.97, "max_rank_ratio": 0.8},
                "default": {"svd_energy": 0.93, "max_rank_ratio": 0.55},
            },
        )

    @classmethod
    def max_compression(cls) -> "AdaptiveCompressor":
        return cls(
            error_threshold=0.10,
            layer_configs={
                "attn": {"svd_energy": 0.85, "max_rank_ratio": 0.3},
                "mlp": {"svd_energy": 0.88, "max_rank_ratio": 0.4},
                "embed": {"svd_energy": 0.92, "max_rank_ratio": 0.5},
                "default": {"svd_energy": 0.86, "max_rank_ratio": 0.35},
            },
        )

    def get_layer_config(self, layer_name: str) -> dict:
        name_lower = layer_name.lower()
        if any(k in name_lower for k in ["attn", "attention", "query", "key", "value"]):
            return self.layer_configs["attn"]
        elif any(k in name_lower for k in ["mlp", "ffn", "fc", "dense"]):
            return self.layer_configs["mlp"]
        elif any(k in name_lower for k in ["embed", "wte", "wpe"]):
            return self.layer_configs["embed"]
        return self.layer_configs["default"]

    def compute_adaptive_rank(
        self,
        weight: torch.Tensor,
        energy_threshold: float,
        max_rank_ratio: float,
    ) -> int:
        u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
        
        total_energy = torch.sum(s ** 2)
        cumulative = torch.cumsum(s ** 2, dim=0) / total_energy
        
        energy_rank = (cumulative < energy_threshold).sum().item() + 1
        max_rank = int(min(weight.shape) * max_rank_ratio)
        
        return min(energy_rank, max_rank)

    def compress_weight(
        self,
        weight: torch.Tensor,
        layer_name: str = "default",
    ) -> dict:
        config = self.get_layer_config(layer_name)
        weight = weight.float()
        original_shape = weight.shape
        
        if weight.dim() == 1:
            return self._passthrough(weight, original_shape)
        
        try:
            rank = self.compute_adaptive_rank(
                weight,
                config["svd_energy"],
                config["max_rank_ratio"],
            )
            
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            
            reconstructed = u @ torch.diag(s) @ vh
            error = torch.linalg.norm(weight - reconstructed) / torch.linalg.norm(weight)
            
            residual = None
            if error > self.error_threshold:
                residual = weight - reconstructed
            
            original_size = weight.numel() * 4
            compressed_size = (u.numel() + s.numel() + vh.numel()) * 4
            if residual is not None:
                compressed_size += residual.numel() * 4
            
            if self.quantize_patterns:
                u_q, u_scale = self._quantize_int8(u)
                vh_q, vh_scale = self._quantize_int8(vh)
                compressed_size = (u.numel() + vh.numel()) + s.numel() * 4 + 8
            else:
                u_q, u_scale = u, None
                vh_q, vh_scale = vh, None
            
            return {
                "type": "svd",
                "U": u_q,
                "U_scale": u_scale,
                "S": s,
                "Vh": vh_q,
                "Vh_scale": vh_scale,
                "residual": residual,
                "rank": rank,
                "original_shape": original_shape,
                "error": error.item(),
                "compression_ratio": original_size / max(compressed_size, 1),
                "size_bytes": compressed_size,
            }
            
        except Exception as e:
            return self._passthrough(weight, original_shape, str(e))

    def decompress_weight(self, compressed: dict) -> torch.Tensor:
        if compressed["type"] == "passthrough":
            return compressed["weight"]
        
        u = compressed["U"]
        s = compressed["S"]
        vh = compressed["Vh"]
        
        if compressed.get("U_scale") is not None:
            u = self._dequantize_int8(u, compressed["U_scale"])
            vh = self._dequantize_int8(vh, compressed["Vh_scale"])
        
        result = u @ torch.diag(s) @ vh
        
        if compressed.get("residual") is not None:
            result = result + compressed["residual"]
        
        return result

    def _quantize_int8(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        max_val = tensor.abs().max()
        scale = max_val / 127.0
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
        return quantized, scale

    def _dequantize_int8(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return quantized.float() * scale

    def _passthrough(self, weight: torch.Tensor, shape: tuple, error: str = None) -> dict:
        return {
            "type": "passthrough",
            "weight": weight,
            "original_shape": shape,
            "error": 0.0 if error is None else error,
            "compression_ratio": 1.0,
            "size_bytes": weight.numel() * 4,
        }

    def compress_model(
        self,
        weights: dict[str, torch.Tensor],
        verbose: bool = True,
    ) -> dict[str, dict]:
        results = {}
        total_original = 0
        total_compressed = 0
        
        for name, weight in weights.items():
            result = self.compress_weight(weight, name)
            results[name] = result
            
            original = weight.numel() * 4
            total_original += original
            total_compressed += result["size_bytes"]
            
            if verbose:
                ratio = result["compression_ratio"]
                error = result["error"] if isinstance(result["error"], float) else 0
                print(f"  {name}: {ratio:.2f}x compression, {error:.2%} error")
        
        if verbose:
            overall_ratio = total_original / total_compressed
            print(f"\nOverall: {overall_ratio:.2f}x compression")
            print(f"  Original: {total_original / 1024 / 1024:.2f} MB")
            print(f"  Compressed: {total_compressed / 1024 / 1024:.2f} MB")
        
        return results

    def decompress_model(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        return {name: self.decompress_weight(data) for name, data in compressed.items()}
