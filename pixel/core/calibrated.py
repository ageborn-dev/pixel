from __future__ import annotations
from typing import Optional, Callable
import torch
import torch.nn as nn
from collections import defaultdict


class ActivationTracker:
    def __init__(self):
        self.activations: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.hooks: list = []

    def register_hooks(self, model: nn.Module, layer_types: tuple = (nn.Linear,)):
        for name, module in model.named_modules():
            if isinstance(module, layer_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str) -> Callable:
        def hook(module, input, output):
            if len(input) > 0 and isinstance(input[0], torch.Tensor):
                self.activations[name].append(input[0].detach().cpu())
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def compute_importance(self) -> dict[str, torch.Tensor]:
        importance = {}
        for name, acts in self.activations.items():
            if acts:
                stacked = torch.cat([a.view(-1, a.shape[-1]) for a in acts], dim=0)
                importance[name] = stacked.abs().mean(dim=0)
        return importance

    def clear(self):
        self.activations.clear()


class CalibratedCompressor:
    def __init__(
        self,
        protection_ratio: float = 0.01,
        base_svd_energy: float = 0.85,
        protected_svd_energy: float = 0.99,
        quantize: bool = True,
    ):
        self.protection_ratio = protection_ratio
        self.base_svd_energy = base_svd_energy
        self.protected_svd_energy = protected_svd_energy
        self.quantize = quantize
        
        self.weight_importance: dict[str, torch.Tensor] = {}
        self.critical_indices: dict[str, torch.Tensor] = {}

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: list[torch.Tensor],
        tokenizer=None,
    ):
        print("Running calibration...")
        tracker = ActivationTracker()
        tracker.register_hooks(model)
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                if isinstance(data, str) and tokenizer:
                    data = tokenizer(data, return_tensors="pt").input_ids
                
                if hasattr(data, "to"):
                    device = next(model.parameters()).device
                    data = data.to(device)
                
                try:
                    model(data)
                except Exception as e:
                    print(f"  Sample {i} failed: {e}")
                    continue
                
                if i % 10 == 0:
                    print(f"  Processed {i+1}/{len(calibration_data)} samples")
        
        tracker.remove_hooks()
        
        self.weight_importance = tracker.compute_importance()
        print(f"Computed importance for {len(self.weight_importance)} layers")
        
        self._identify_critical_weights()
        tracker.clear()

    def _identify_critical_weights(self):
        for name, importance in self.weight_importance.items():
            num_critical = max(1, int(len(importance) * self.protection_ratio))
            _, indices = torch.topk(importance, num_critical)
            self.critical_indices[name] = indices
        
        total_critical = sum(len(idx) for idx in self.critical_indices.values())
        total_weights = sum(len(imp) for imp in self.weight_importance.values())
        print(f"Identified {total_critical}/{total_weights} critical weight columns ({100*total_critical/total_weights:.2f}%)")

    def compress_weight(
        self,
        weight: torch.Tensor,
        layer_name: str,
    ) -> dict:
        weight = weight.float()
        original_shape = weight.shape
        
        if weight.dim() == 1:
            return self._passthrough(weight, original_shape)
        
        is_protected = layer_name in self.critical_indices
        svd_energy = self.protected_svd_energy if is_protected else self.base_svd_energy
        
        try:
            u, s, vh = torch.linalg.svd(weight, full_matrices=False)
            
            total_energy = torch.sum(s ** 2)
            cumulative = torch.cumsum(s ** 2, dim=0) / total_energy
            rank = (cumulative < svd_energy).sum().item() + 1
            
            u = u[:, :rank]
            s = s[:rank]
            vh = vh[:rank, :]
            
            reconstructed = u @ torch.diag(s) @ vh
            error = torch.linalg.norm(weight - reconstructed) / torch.linalg.norm(weight)
            
            residual = None
            if is_protected and layer_name in self.critical_indices:
                critical_cols = self.critical_indices[layer_name]
                residual_cols = weight[:, critical_cols] - reconstructed[:, critical_cols]
                if residual_cols.abs().mean() > 0.01:
                    residual = {"cols": critical_cols, "values": residual_cols}
            
            original_size = weight.numel() * 4
            compressed_size = (u.numel() + s.numel() + vh.numel())
            
            if self.quantize:
                u_q, u_scale = self._quantize_int8(u)
                vh_q, vh_scale = self._quantize_int8(vh)
                compressed_size = u.numel() + vh.numel() + s.numel() * 4 + 8
            else:
                u_q, u_scale = u, None
                vh_q, vh_scale = vh, None
                compressed_size = compressed_size * 4
            
            if residual:
                compressed_size += residual["values"].numel() * 4 + len(residual["cols"]) * 4
            
            return {
                "type": "calibrated_svd",
                "U": u_q,
                "U_scale": u_scale,
                "S": s,
                "Vh": vh_q,
                "Vh_scale": vh_scale,
                "residual": residual,
                "rank": rank,
                "protected": is_protected,
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
            residual = compressed["residual"]
            result[:, residual["cols"]] = result[:, residual["cols"]] + residual["values"]
        
        return result

    def compress_model(
        self,
        weights: dict[str, torch.Tensor],
        verbose: bool = True,
    ) -> dict[str, dict]:
        if not self.weight_importance:
            print("WARNING: No calibration data. Run calibrate() first for best results.")
        
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
                protected = "🛡️" if result.get("protected") else "  "
                error = result["error"] if isinstance(result["error"], float) else 0
                ratio = result["compression_ratio"]
                print(f"{protected} {name}: {ratio:.2f}x, {error:.2%} error")
        
        if verbose:
            overall_ratio = total_original / total_compressed
            print(f"\nOverall: {overall_ratio:.2f}x compression")
            print(f"  Original: {total_original / 1024 / 1024:.2f} MB")
            print(f"  Compressed: {total_compressed / 1024 / 1024:.2f} MB")
        
        return results

    def decompress_model(self, compressed: dict[str, dict]) -> dict[str, torch.Tensor]:
        return {name: self.decompress_weight(data) for name, data in compressed.items()}

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
            "protected": False,
        }
