"""Basic usage example for PIXEL compression."""
import torch
from pixel import PIXELConfig, PIXELCompressor


def main():
    print("PIXEL Basic Usage Example")
    print("=" * 50)

    config = PIXELConfig.for_small_model()
    compressor = PIXELCompressor(config)
    print(f"Initialized compressor with {len(compressor.pattern_dict)} base patterns")

    weights = {
        "layer1.weight": torch.randn(64, 64),
        "layer2.weight": torch.randn(64, 64),
        "layer3.weight": torch.randn(64, 64),
    }

    original_size = sum(w.numel() * 4 for w in weights.values())
    print(f"\nOriginal weights: {len(weights)} layers, {original_size / 1024:.2f} KB")

    result = compressor.compress(weights)

    print(f"\nCompression Results:")
    print(f"  - Compression ratio: {result.compression_ratio:.2f}x")
    print(f"  - Reconstruction error: {result.reconstruction_error:.4f}")
    print(f"  - Patterns used: {result.num_patterns}")
    print(f"  - Compressed size: {result.compressed_size / 1024:.2f} KB")

    reconstructed = compressor.decompress(result.pattern_refs)
    print(f"\nReconstructed {len(reconstructed)} weight matrices")

    print("\n" + "=" * 50)
    print("Single tensor compression:")

    tensor = torch.randn(32, 32)
    refs, residual, error = compressor.compress_tensor(tensor)

    print(f"  - Pattern references: {len(refs)}")
    print(f"  - Reconstruction error: {error:.4f}")

    recovered = compressor.decompress_tensor(refs, residual)
    actual_error = torch.linalg.norm(tensor - recovered) / torch.linalg.norm(tensor)
    print(f"  - Verified error: {actual_error:.4f}")


if __name__ == "__main__":
    main()
