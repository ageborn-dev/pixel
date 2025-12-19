"""Compression quality benchmark."""
import torch

from pixel import PIXELConfig, PIXELCompressor
from pixel.experimental import SVDHybridCompressor


def generate_test_matrices(size, num_matrices=10):
    matrices = {}

    matrices["identity"] = torch.eye(size)

    x = torch.linspace(0, 2 * 3.14159, size)
    matrices["fourier_sin"] = torch.outer(torch.sin(x), torch.sin(x))
    matrices["fourier_cos"] = torch.outer(torch.cos(x), torch.cos(x))

    matrices["block_diagonal"] = torch.zeros(size, size)
    block_size = size // 4
    for i in range(4):
        start = i * block_size
        end = start + block_size
        matrices["block_diagonal"][start:end, start:end] = torch.randn(block_size, block_size)

    matrices["low_rank"] = torch.randn(size, 8) @ torch.randn(8, size)

    matrices["sparse"] = torch.zeros(size, size)
    indices = torch.randint(0, size, (size * size // 10, 2))
    for i, j in indices:
        matrices["sparse"][i, j] = torch.randn(1).item()

    for i in range(num_matrices - len(matrices)):
        matrices[f"random_{i}"] = torch.randn(size, size)

    return matrices


def run_accuracy_benchmark():
    print("PIXEL Compression Quality Benchmark")
    print("=" * 70)

    size = 64
    matrices = generate_test_matrices(size)

    print(f"\nTest matrices: {len(matrices)} different patterns, {size}x{size} each")
    print("-" * 70)

    config = PIXELConfig(
        matrix_size=size,
        compression=PIXELConfig.for_small_model().compression,
    )
    compressor = PIXELCompressor(config)
    svd_hybrid = SVDHybridCompressor(svd_energy_threshold=0.9)

    print(f"\n{'Matrix':<20} {'PIXEL Err':<12} {'PIXEL Ratio':<12} {'SVD+P Err':<12} {'SVD+P Ratio':<12}")
    print("-" * 70)

    pixel_errors = []
    svd_errors = []

    for name, matrix in matrices.items():
        p_refs, p_residual, p_error = compressor.compress_tensor(matrix)
        p_recon = compressor.decompress_tensor(p_refs, p_residual)
        p_ratio = matrix.numel() * 4 / (len(p_refs) * 8 + p_residual.numel() * 4)

        s_result = svd_hybrid.compress(matrix)
        s_error = s_result["reconstruction_error"]
        s_ratio = s_result["compression_ratio"]

        pixel_errors.append(p_error)
        svd_errors.append(s_error)

        print(f"{name:<20} {p_error:<12.4f} {p_ratio:<12.2f} {s_error:<12.4f} {s_ratio:<12.2f}")

    print("-" * 70)
    print(f"{'AVERAGE':<20} {sum(pixel_errors)/len(pixel_errors):<12.4f} {'':<12} "
          f"{sum(svd_errors)/len(svd_errors):<12.4f}")

    print("\n" + "=" * 70)
    print("Pattern Efficiency by Max Patterns:")
    print("-" * 70)

    test_matrix = matrices["random_0"]
    print(f"\n{'Max Patterns':<15} {'Error':<12} {'Patterns Used':<15} {'Ratio':<10}")
    print("-" * 50)

    for max_p in [2, 4, 8, 16, 32]:
        refs, residual, error = compressor.compress_tensor(test_matrix, max_patterns=max_p)
        ratio = test_matrix.numel() * 4 / (len(refs) * 8 + residual.numel() * 4)
        print(f"{max_p:<15} {error:<12.4f} {len(refs):<15} {ratio:<10.2f}")


if __name__ == "__main__":
    run_accuracy_benchmark()
