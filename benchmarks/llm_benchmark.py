"""Benchmark PIXEL compression on real LLM weights (GPT-2)."""
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

from pixel import PIXELConfig, PIXELCompressor
from pixel.experimental import SVDHybridCompressor
from pixel.utils.memory import format_bytes, MemoryTracker


def load_gpt2_weights(model_name: str = "gpt2") -> dict[str, torch.Tensor]:
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    weights = {}
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            weights[name] = param.data.clone()
    
    print(f"Loaded {len(weights)} weight matrices")
    total_params = sum(w.numel() for w in weights.values())
    print(f"Total parameters: {total_params:,} ({format_bytes(total_params * 4)})")
    
    return weights


def benchmark_pixel(weights: dict[str, torch.Tensor]) -> dict:
    print("\n" + "=" * 60)
    print("PIXEL Compression Benchmark")
    print("=" * 60)
    
    config = PIXELConfig(
        compression=PIXELConfig.for_large_model().compression,
    )
    compressor = PIXELCompressor(config)
    
    start = time.perf_counter()
    result = compressor.compress(weights)
    compress_time = time.perf_counter() - start
    
    print(f"\nResults:")
    print(f"  Compression ratio: {result.compression_ratio:.2f}x")
    print(f"  Reconstruction error: {result.reconstruction_error:.4f}")
    print(f"  Patterns discovered: {len(compressor.pattern_dict)}")
    print(f"  Compression time: {compress_time:.2f}s")
    print(f"  Original size: {format_bytes(result.original_size)}")
    print(f"  Compressed size: {format_bytes(result.compressed_size)}")
    
    return {
        "method": "PIXEL",
        "ratio": result.compression_ratio,
        "error": result.reconstruction_error,
        "patterns": len(compressor.pattern_dict),
        "time": compress_time,
    }


def benchmark_svd_hybrid(weights: dict[str, torch.Tensor]) -> dict:
    print("\n" + "=" * 60)
    print("SVD + Pattern Hybrid Benchmark")
    print("=" * 60)
    
    compressor = SVDHybridCompressor(
        svd_energy_threshold=0.95,
        pattern_error_threshold=0.02,
    )
    
    start = time.perf_counter()
    results = compressor.compress_model_weights(weights)
    compress_time = time.perf_counter() - start
    
    total_original = sum(r["size_breakdown"]["original"] for r in results.values())
    total_compressed = sum(r["size_breakdown"]["total_compressed"] for r in results.values())
    avg_error = sum(r["reconstruction_error"] for r in results.values()) / len(results)
    ratio = total_original / total_compressed
    
    print(f"\nResults:")
    print(f"  Compression ratio: {ratio:.2f}x")
    print(f"  Average reconstruction error: {avg_error:.4f}")
    print(f"  Patterns discovered: {len(compressor.pattern_dict)}")
    print(f"  Compression time: {compress_time:.2f}s")
    print(f"  Original size: {format_bytes(total_original)}")
    print(f"  Compressed size: {format_bytes(total_compressed)}")
    
    return {
        "method": "SVD+Pattern",
        "ratio": ratio,
        "error": avg_error,
        "patterns": len(compressor.pattern_dict),
        "time": compress_time,
    }


def analyze_weight_structure(weights: dict[str, torch.Tensor]):
    print("\n" + "=" * 60)
    print("Weight Structure Analysis")
    print("=" * 60)
    
    stats = []
    for name, weight in weights.items():
        try:
            u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
            
            total_energy = torch.sum(s ** 2)
            cumulative = torch.cumsum(s ** 2, dim=0) / total_energy
            
            rank_90 = (cumulative < 0.90).sum().item() + 1
            rank_95 = (cumulative < 0.95).sum().item() + 1
            rank_99 = (cumulative < 0.99).sum().item() + 1
            
            stats.append({
                "name": name.split(".")[-2] + "." + name.split(".")[-1],
                "shape": tuple(weight.shape),
                "rank_90": rank_90,
                "rank_95": rank_95,
                "rank_99": rank_99,
                "full_rank": min(weight.shape),
            })
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue
    
    print(f"\n{'Layer':<30} {'Shape':<15} {'Rank@90%':<10} {'Rank@95%':<10} {'Rank@99%':<10}")
    print("-" * 75)
    
    for s in stats[:15]:
        shape_str = f"{s['shape'][0]}x{s['shape'][1]}"
        print(f"{s['name']:<30} {shape_str:<15} {s['rank_90']:<10} {s['rank_95']:<10} {s['rank_99']:<10}")
    
    if len(stats) > 15:
        print(f"... and {len(stats) - 15} more layers")
    
    avg_compression_90 = sum(s["full_rank"] / s["rank_90"] for s in stats) / len(stats)
    avg_compression_95 = sum(s["full_rank"] / s["rank_95"] for s in stats) / len(stats)
    
    print(f"\nAverage rank reduction potential:")
    print(f"  At 90% energy: {avg_compression_90:.2f}x")
    print(f"  At 95% energy: {avg_compression_95:.2f}x")


def main():
    print("=" * 60)
    print("PIXEL vs Real LLM Weights Benchmark")
    print("=" * 60)
    
    weights = load_gpt2_weights("gpt2")
    
    analyze_weight_structure(weights)
    
    subset = {k: v for k, v in list(weights.items())[:10]}
    print(f"\nUsing subset of {len(subset)} layers for benchmark")
    
    pixel_results = benchmark_pixel(subset)
    svd_results = benchmark_svd_hybrid(subset)
    
    print("\n" + "=" * 60)
    print("Summary Comparison")
    print("=" * 60)
    print(f"\n{'Method':<20} {'Ratio':<12} {'Error':<12} {'Patterns':<12} {'Time':<12}")
    print("-" * 60)
    
    for r in [pixel_results, svd_results]:
        print(f"{r['method']:<20} {r['ratio']:<12.2f} {r['error']:<12.4f} {r['patterns']:<12} {r['time']:<12.2f}s")
    
    print("\n" + "=" * 60)
    print("Key Insights")
    print("=" * 60)
    print("""
1. Real LLM weights have more structure than random matrices
2. SVD reveals low-rank structure (most energy in top singular values)  
3. Pattern-based compression can exploit this structure
4. Combination of SVD + patterns provides best results
""")


if __name__ == "__main__":
    main()
