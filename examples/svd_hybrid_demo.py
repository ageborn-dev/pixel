"""Example: SVD + Pattern hybrid compression."""
import torch

from pixel.experimental import SVDHybridCompressor
from pixel import PIXELConfig


def main():
    print("SVD + Pattern Hybrid Compression Example")
    print("=" * 50)

    compressor = SVDHybridCompressor(
        svd_energy_threshold=0.85,
        pattern_error_threshold=0.05,
    )

    print(f"Compressor stats: {compressor.stats()}")

    weight = torch.randn(64, 64)
    print(f"\nOriginal weight: {weight.shape}, {weight.numel() * 4} bytes")

    result = compressor.compress(weight)

    print(f"\nCompression results:")
    print(f"  - SVD rank: {result['svd_rank']}")
    print(f"  - Patterns used: {result['num_patterns']}")
    print(f"  - Reconstruction error: {result['reconstruction_error']:.4f}")
    print(f"  - Compression ratio: {result['compression_ratio']:.2f}x")

    print(f"\nSize breakdown:")
    breakdown = result['size_breakdown']
    print(f"  - Original: {breakdown['original']:,} bytes")
    print(f"  - SVD components: {breakdown['svd']:,} bytes")
    print(f"  - Pattern refs: {breakdown['patterns']:,} bytes")
    print(f"  - Residual: {breakdown['residual']:,} bytes")
    print(f"  - Total compressed: {breakdown['total_compressed']:,} bytes")

    reconstructed = compressor.decompress(result)
    actual_error = torch.linalg.norm(weight - reconstructed) / torch.linalg.norm(weight)
    print(f"\nVerified reconstruction error: {actual_error:.4f}")

    print("\n" + "=" * 50)
    print("Batch compression example:")

    weights = {
        "attention.query": torch.randn(256, 256),
        "attention.key": torch.randn(256, 256),
        "attention.value": torch.randn(256, 256),
        "ffn.up": torch.randn(256, 1024),
        "ffn.down": torch.randn(1024, 256),
    }

    compressed_all = compressor.compress_model_weights(weights)

    total_original = sum(w.numel() * 4 for w in weights.values())
    total_compressed = sum(
        r['size_breakdown']['total_compressed'] for r in compressed_all.values()
    )
    avg_error = sum(r['reconstruction_error'] for r in compressed_all.values()) / len(compressed_all)

    print(f"\nBatch results:")
    print(f"  - Layers: {len(weights)}")
    print(f"  - Original size: {total_original / 1024:.2f} KB")
    print(f"  - Compressed size: {total_compressed / 1024:.2f} KB")
    print(f"  - Overall ratio: {total_original / total_compressed:.2f}x")
    print(f"  - Average error: {avg_error:.4f}")


if __name__ == "__main__":
    main()
