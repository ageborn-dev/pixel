"""Memory usage benchmark for PIXEL."""
import time
import torch

from pixel import PIXELConfig, PIXELCompressor
from pixel.utils.memory import MemoryTracker, format_bytes


def run_memory_benchmark():
    print("PIXEL Memory Benchmark")
    print("=" * 60)

    tracker = MemoryTracker()
    tracker.snapshot("Initial")

    sizes = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
    results = []

    for rows, cols in sizes:
        print(f"\nTesting {rows}x{cols} matrices...")

        config = PIXELConfig(
            matrix_size=max(rows, cols),
            compression=PIXELConfig.for_small_model().compression,
        )
        compressor = PIXELCompressor(config)

        tracker.snapshot(f"After compressor init ({rows}x{cols})")

        weights = {f"layer_{i}": torch.randn(rows, cols) for i in range(4)}
        original_size = sum(w.numel() * 4 for w in weights.values())

        tracker.snapshot(f"After creating weights ({rows}x{cols})")

        start = time.perf_counter()
        result = compressor.compress(weights)
        compress_time = time.perf_counter() - start

        tracker.snapshot(f"After compression ({rows}x{cols})")

        start = time.perf_counter()
        _ = compressor.decompress(result.pattern_refs)
        decompress_time = time.perf_counter() - start

        results.append({
            "size": f"{rows}x{cols}",
            "original_bytes": original_size,
            "compressed_bytes": result.compressed_size,
            "ratio": result.compression_ratio,
            "error": result.reconstruction_error,
            "compress_ms": compress_time * 1000,
            "decompress_ms": decompress_time * 1000,
        })

        tracker.snapshot(f"After decompression ({rows}x{cols})")

        del weights, compressor
        MemoryTracker.clear_cuda_cache()

    print("\n" + "=" * 60)
    print("Results Summary:")
    print("-" * 60)
    print(f"{'Size':<12} {'Original':<12} {'Compressed':<12} {'Ratio':<8} {'Error':<8} {'Time(ms)':<10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['size']:<12} "
            f"{format_bytes(r['original_bytes']):<12} "
            f"{format_bytes(r['compressed_bytes']):<12} "
            f"{r['ratio']:<8.2f} "
            f"{r['error']:<8.4f} "
            f"{r['compress_ms']:<10.1f}"
        )

    print("\n" + tracker.report())

    return results


if __name__ == "__main__":
    run_memory_benchmark()
