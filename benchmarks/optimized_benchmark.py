"""Optimized benchmark with AdaptiveCompressor on GPT-2."""
import torch
from transformers import GPT2LMHeadModel

from pixel.core.adaptive import AdaptiveCompressor
from pixel.utils.memory import format_bytes


def load_gpt2_weights(model_name: str = "gpt2") -> dict[str, torch.Tensor]:
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    weights = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            weights[name] = param.data.clone()
    
    print(f"Loaded {len(weights)} weight matrices")
    total_params = sum(w.numel() for w in weights.values())
    print(f"Total: {total_params:,} parameters ({format_bytes(total_params * 4)})")
    
    return weights


def run_optimized_benchmark():
    print("=" * 60)
    print("PIXEL Optimized Benchmark (High Quality Preset)")
    print("=" * 60)
    
    compressor = AdaptiveCompressor.high_quality()
    
    print(f"\nSettings (high_quality):")
    print(f"  Error threshold: {compressor.error_threshold:.0%}")
    print(f"  Quantize patterns: {compressor.quantize_patterns}")
    print(f"  Layer configs: {compressor.layer_configs}")
    
    weights = load_gpt2_weights()
    
    print("\n" + "-" * 60)
    print("Compressing layers...")
    print("-" * 60)
    
    results = compressor.compress_model(weights, verbose=True)
    
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    decompressed = compressor.decompress_model(results)
    
    errors = []
    for name in list(weights.keys())[:5]:
        if name in decompressed:
            original = weights[name]
            recovered = decompressed[name]
            error = torch.linalg.norm(original - recovered) / torch.linalg.norm(original)
            errors.append(error.item())
            print(f"  {name.split('.')[-1]}: verified error = {error:.4f}")
    
    print(f"\nAverage verified error: {sum(errors)/len(errors):.4f}")
    
    total_original = sum(w.numel() * 4 for w in weights.values())
    total_compressed = sum(r["size_bytes"] for r in results.values())
    
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"  Original size: {format_bytes(total_original)}")
    print(f"  Compressed size: {format_bytes(total_compressed)}")
    print(f"  Overall ratio: {total_original / total_compressed:.2f}x")
    
    by_type = {"attn": [], "mlp": [], "embed": [], "other": []}
    for name, result in results.items():
        if "attn" in name.lower():
            by_type["attn"].append(result["compression_ratio"])
        elif "mlp" in name.lower() or "fc" in name.lower():
            by_type["mlp"].append(result["compression_ratio"])
        elif "embed" in name.lower() or "wte" in name.lower():
            by_type["embed"].append(result["compression_ratio"])
        else:
            by_type["other"].append(result["compression_ratio"])
    
    print("\nBy layer type:")
    for ltype, ratios in by_type.items():
        if ratios:
            avg = sum(ratios) / len(ratios)
            print(f"  {ltype}: {avg:.2f}x average ({len(ratios)} layers)")


if __name__ == "__main__":
    run_optimized_benchmark()
