from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn


class GPTQStyleCompressor:
    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        svd_rank_ratio: float = 0.5,
        use_svd: bool = True,
        dampening: float = 0.01,
    ):
        self.bits = bits
        self.group_size = group_size
        self.svd_rank_ratio = svd_rank_ratio
        self.use_svd = use_svd
        self.dampening = dampening

    def quantize_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        max_int = 2 ** self.bits - 1
        quantized = torch.clamp(
            torch.round(weight / scale + zero_point),
            0, max_int
        )
        return quantized.to(torch.int8)

    def dequantize_weight(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        return (quantized.float() - zero_point) * scale

    def compute_scale_zp(
        self,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_int = 2 ** self.bits - 1
        
        w_min = weight.min(dim=-1, keepdim=True).values
        w_max = weight.max(dim=-1, keepdim=True).values
        
        scale = (w_max - w_min) / max_int
        scale = torch.clamp(scale, min=1e-8)
        zero_point = torch.round(-w_min / scale)
        
        return scale, zero_point

    def compress_layer_gptq_style(
        self,
        weight: torch.Tensor,
        hessian: Optional[torch.Tensor] = None,
    ) -> dict:
        weight = weight.float()
        rows, cols = weight.shape
        
        if hessian is None:
            hessian = torch.eye(cols, device=weight.device, dtype=weight.dtype)
        
        hessian = hessian + self.dampening * torch.eye(cols, device=weight.device)
        
        try:
            hessian_inv = torch.linalg.inv(hessian)
        except:
            hessian_inv = torch.eye(cols, device=weight.device)
        
        quantized_weight = torch.zeros_like(weight)
        
        for col in range(cols):
            w_col = weight[:, col].clone()
            
            w_min = w_col.min()
            w_max = w_col.max()
            max_int = 2 ** self.bits - 1
            scale = (w_max - w_min) / max_int
            scale = max(scale.item(), 1e-8)
            zero_point = round((-w_min / scale).item())
            
            q_col = torch.clamp(torch.round(w_col / scale + zero_point), 0, max_int)
            dq_col = (q_col - zero_point) * scale
            
            quantized_weight[:, col] = dq_col
            
            error = w_col - dq_col
            
            if col < cols - 1:
                h_diag = hessian_inv[col, col].item()
                if h_diag > 1e-8:
                    compensation = error.unsqueeze(1) * hessian_inv[col, col+1:].unsqueeze(0) / h_diag
                    weight[:, col+1:] = weight[:, col+1:] + compensation
        
        original_size = weight.numel() * 4
        compressed_size = weight.numel() * self.bits / 8 + rows * 8
        
        return {
            "type": "gptq_style",
            "quantized": quantized_weight,
            "original_shape": weight.shape,
            "bits": self.bits,
            "compression_ratio": original_size / compressed_size,
            "size_bytes": int(compressed_size),
        }

    def compute_hessian_approx(
        self,
        layer: nn.Linear,
        calibration_inputs: list[torch.Tensor],
    ) -> torch.Tensor:
        device = layer.weight.device
        in_features = layer.in_features
        
        hessian = torch.zeros(in_features, in_features, device=device)
        n_samples = 0
        
        for inp in calibration_inputs:
            if inp.dim() == 3:
                inp = inp.view(-1, inp.shape[-1])
            
            if inp.shape[-1] == in_features:
                inp = inp.to(device)
                hessian = hessian + inp.T @ inp
                n_samples += inp.shape[0]
        
        if n_samples > 0:
            hessian = hessian / n_samples
        else:
            hessian = torch.eye(in_features, device=device)
        
        return hessian

    def calibrate(
        self,
        model: nn.Module,
        calibration_data: list[torch.Tensor],
        verbose: bool = True,
    ) -> dict[str, torch.Tensor]:
        layer_activations = {}
        hooks = []
        
        def make_hook(name):
            def hook(module, inp, out):
                if len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if x.dim() == 3:
                        x = x.view(-1, x.shape[-1])
                    if name not in layer_activations:
                        layer_activations[name] = []
                    layer_activations[name].append(x.cpu())
            return hook
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append((name, module.register_forward_hook(make_hook(name))))
        
        if verbose:
            print(f"Running {len(calibration_data)} calibration samples...")
        
        model.eval()
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for i, data in enumerate(calibration_data):
                try:
                    if hasattr(data, "to"):
                        data = data.to(device)
                    model(data)
                except Exception as e:
                    pass
        
        for name, hook in hooks:
            hook.remove()
        
        hessians = {}
        for name, acts in layer_activations.items():
            if acts:
                try:
                    stacked = torch.cat(acts, dim=0).float()
                    n_samples = stacked.shape[0]
                    hessian = (stacked.T @ stacked) / n_samples
                    hessians[name] = hessian
                except:
                    pass
        
        if verbose:
            print(f"Computed Hessians for {len(hessians)} layers")
        
        self.hessians = hessians
        return hessians

    def compress_model_sequential(
        self,
        model: nn.Module,
        calibration_data: list[torch.Tensor] = None,
        verbose: bool = True,
    ) -> dict[str, dict]:
        hessians = {}
        if calibration_data:
            hessians = self.calibrate(model, calibration_data, verbose)
        
        results = {}
        total_original = 0
        total_compressed = 0
        
        state_dict = model.state_dict()
        
        name_to_module = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_name = name + ".weight"
                name_to_module[weight_name] = name
        
        weight_names = [name for name in state_dict.keys() if "weight" in name and state_dict[name].dim() == 2]
        
        if verbose:
            print(f"Compressing {len(weight_names)} weight matrices...")
        
        for weight_name in weight_names:
            weight = state_dict[weight_name].clone()
            
            hessian = None
            module_name = name_to_module.get(weight_name)
            if module_name and module_name in hessians:
                h = hessians[module_name]
                if h.shape[0] == weight.shape[1]:
                    hessian = h.to(weight.device)
            
            result = self.compress_layer_gptq_style(weight, hessian)
            results[weight_name] = result
            
            total_original += weight.numel() * 4
            total_compressed += result["size_bytes"]
            
            if verbose:
                recon_error = torch.linalg.norm(weight - result["quantized"]) / torch.linalg.norm(weight)
                short_name = weight_name.split(".")[-2] + "." + weight_name.split(".")[-1] if "." in weight_name else weight_name
                hess_status = "✓" if hessian is not None else " "
                print(f"  {hess_status} {short_name}: {result['compression_ratio']:.2f}x, {recon_error:.2%} error")
        
        if verbose:
            print(f"\nOverall: {total_original/total_compressed:.2f}x compression")
            print(f"  Original: {total_original/1024/1024:.2f} MB")
            print(f"  Compressed: {total_compressed/1024/1024:.2f} MB")
        
        return results

    def apply_compressed_weights(
        self,
        model: nn.Module,
        compressed: dict[str, dict],
    ) -> nn.Module:
        state_dict = model.state_dict()
        
        for name, result in compressed.items():
            if name in state_dict:
                state_dict[name] = result["quantized"].to(state_dict[name].dtype)
        
        model.load_state_dict(state_dict)
        return model
