"""Integration tests for PIXEL."""
import pytest
import torch
import torch.nn as nn

from pixel import PIXELConfig, PIXELCompressor, PatternDictionary
from pixel.nn.layers import PIXELLinear, convert_linear_to_pixel, replace_linear_layers
from pixel.experimental import SVDHybridCompressor


class TestEndToEndWorkflow:
    def test_basic_compression_workflow(self):
        compressor = PIXELCompressor()

        weights = {
            "encoder.weight": torch.randn(64, 64),
            "decoder.weight": torch.randn(64, 64),
        }

        result = compressor.compress(weights)
        assert result.compression_ratio > 1.0
        assert result.reconstruction_error < 1.0

        reconstructed = compressor.decompress(result.pattern_refs)
        assert len(reconstructed) == len(weights)

    def test_linear_layer_conversion(self):
        pattern_dict = PatternDictionary(max_patterns=50)
        pattern_dict.generate_base_patterns(16)

        linear = nn.Linear(16, 16)
        pixel_layer, error = convert_linear_to_pixel(linear, pattern_dict)

        assert isinstance(pixel_layer, PIXELLinear)
        assert error < 1.0

        x = torch.randn(4, 16)
        output = pixel_layer(x)
        assert output.shape == (4, 16)

    def test_model_replacement(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(16, 32)
                self.fc2 = nn.Linear(32, 16)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = SimpleModel()
        pattern_dict = PatternDictionary(max_patterns=100)
        pattern_dict.generate_base_patterns(16)
        pattern_dict.generate_base_patterns(32)

        errors = replace_linear_layers(model, pattern_dict)
        
        assert len(errors) == 2
        assert isinstance(model.fc1, PIXELLinear)
        assert isinstance(model.fc2, PIXELLinear)

        x = torch.randn(4, 16)
        output = model(x)
        assert output.shape == (4, 16)


class TestSVDHybrid:
    def test_svd_hybrid_compression(self):
        compressor = SVDHybridCompressor()
        weight = torch.randn(32, 32)

        result = compressor.compress(weight)

        assert "svd" in result
        assert "pattern_refs" in result
        assert result["compression_ratio"] > 1.0

    def test_svd_hybrid_roundtrip(self):
        compressor = SVDHybridCompressor(
            svd_energy_threshold=0.9,
            pattern_error_threshold=0.05,
        )
        weight = torch.randn(32, 32)

        compressed = compressor.compress(weight)
        reconstructed = compressor.decompress(compressed)

        error = torch.linalg.norm(weight - reconstructed) / torch.linalg.norm(weight)
        assert error < 0.3


class TestConfigPresets:
    def test_small_model_preset(self):
        config = PIXELConfig.for_small_model()
        compressor = PIXELCompressor(config)

        tensor = torch.randn(64, 64)
        refs, residual, error = compressor.compress_tensor(tensor)
        assert error < 0.5

    def test_large_model_preset(self):
        config = PIXELConfig.for_large_model()
        assert config.matrix_size == 512
        assert config.compression.max_patterns_per_layer == 32
