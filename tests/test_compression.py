"""Unit tests for compression pipeline."""
import pytest
import torch

from pixel.core.config import PIXELConfig, CompressionConfig
from pixel.core.compression import PIXELCompressor, CompressionResult


class TestPIXELCompressor:
    def setup_method(self):
        config = PIXELConfig(
            matrix_size=16,
            compression=CompressionConfig(
                pattern_sizes=(4, 8, 16),
                max_patterns_per_layer=8,
            ),
        )
        self.compressor = PIXELCompressor(config)

    def test_compress_single_tensor(self):
        tensor = torch.randn(16, 16)
        refs, residual, error = self.compressor.compress_tensor(tensor)

        assert isinstance(refs, list)
        assert isinstance(residual, torch.Tensor)
        assert 0 <= error <= 1

    def test_decompress_tensor(self):
        tensor = torch.randn(16, 16)
        refs, residual, _ = self.compressor.compress_tensor(tensor)

        reconstructed = self.compressor.decompress_tensor(refs, residual)
        assert reconstructed is not None
        assert reconstructed.shape == tensor.shape

    def test_roundtrip_quality(self):
        tensor = torch.randn(16, 16)
        refs, residual, error = self.compressor.compress_tensor(tensor)
        reconstructed = self.compressor.decompress_tensor(refs, residual)

        actual_error = torch.linalg.norm(tensor - reconstructed) / torch.linalg.norm(tensor)
        assert actual_error < 0.2

    def test_compress_weight_dict(self):
        weights = {
            "layer1.weight": torch.randn(16, 16),
            "layer2.weight": torch.randn(16, 16),
        }

        result = self.compressor.compress(weights)

        assert isinstance(result, CompressionResult)
        assert result.compression_ratio > 0
        assert len(result.pattern_refs) == 2

    def test_decompress_weight_dict(self):
        weights = {
            "layer1.weight": torch.randn(16, 16),
        }

        result = self.compressor.compress(weights)
        reconstructed = self.compressor.decompress(result.pattern_refs)

        assert "layer1.weight" in reconstructed

    def test_progress_callback(self):
        weights = {"layer": torch.randn(16, 16)}
        progress_values = []

        def callback(name, progress):
            progress_values.append(progress)

        self.compressor.compress(weights, progress_callback=callback)
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0


class TestCompressionResult:
    def test_summary_format(self):
        result = CompressionResult(
            original_size=1000,
            compressed_size=100,
            num_patterns=5,
            reconstruction_error=0.05,
            compression_ratio=10.0,
            pattern_refs={},
        )

        summary = result.summary()
        assert "10.00x" in summary
        assert "0.0500" in summary


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        config = PIXELConfig(matrix_size=8)
        compressor = PIXELCompressor(config)

        tensor = torch.randn(8, 8)
        compressor.compress_tensor(tensor)

        save_path = tmp_path / "compressor.pt"
        compressor.save(str(save_path))

        loaded = PIXELCompressor.load(str(save_path))
        assert len(loaded.pattern_dict) == len(compressor.pattern_dict)
