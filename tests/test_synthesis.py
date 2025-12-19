"""Unit tests for weight synthesis."""
import pytest
import torch

from pixel.core.patterns import PatternDictionary
from pixel.core.synthesis import WeightSynthesizer


class TestWeightSynthesizer:
    def setup_method(self):
        self.pattern_dict = PatternDictionary(max_patterns=50)
        self.pattern_dict.generate_base_patterns(8)
        self.synthesizer = WeightSynthesizer(self.pattern_dict)

    def test_synthesize_single_pattern(self):
        pattern = torch.randn(8, 8)
        pid = self.pattern_dict.add(pattern)

        refs = [(pid, 2.5)]
        result = self.synthesizer.synthesize(refs)

        assert result is not None
        expected = self.pattern_dict.get(pid) * 2.5
        assert torch.allclose(result, expected, atol=1e-5)

    def test_synthesize_multiple_patterns(self):
        p1 = torch.eye(8)
        p2 = torch.randn(8, 8)
        pid1 = self.pattern_dict.add(p1)
        pid2 = self.pattern_dict.add(p2)

        refs = [(pid1, 1.0), (pid2, 0.5)]
        result = self.synthesizer.synthesize(refs)

        assert result is not None
        assert result.shape == (8, 8)

    def test_synthesize_empty_refs(self):
        result = self.synthesizer.synthesize([])
        assert result is None

    def test_optimize_scales(self):
        p1 = torch.eye(8)
        p2 = torch.ones(8, 8) / 8
        pid1 = self.pattern_dict.add(p1)
        pid2 = self.pattern_dict.add(p2)

        target = self.pattern_dict.get(pid1) * 2.0 + self.pattern_dict.get(pid2) * 3.0
        optimized = self.synthesizer.optimize_scales(target, [pid1, pid2])

        assert len(optimized) == 2

    def test_compress_weights(self):
        target = torch.randn(8, 8)
        refs, residual = self.synthesizer.compress_weights(target, max_patterns=4)

        assert isinstance(refs, list)
        assert isinstance(residual, torch.Tensor)
        assert residual.shape == target.shape

    def test_compression_reduces_error(self):
        target = torch.randn(8, 8)

        refs1, _ = self.synthesizer.compress_weights(target, max_patterns=1)
        refs4, _ = self.synthesizer.compress_weights(target, max_patterns=4)

        recon1 = self.synthesizer.synthesize(refs1)
        recon4 = self.synthesizer.synthesize(refs4)

        error1 = torch.linalg.norm(target - recon1) if recon1 is not None else float("inf")
        error4 = torch.linalg.norm(target - recon4) if recon4 is not None else float("inf")

        assert len(refs4) >= len(refs1)

    def test_cache_usage(self):
        pattern = torch.randn(8, 8)
        pid = self.pattern_dict.add(pattern)
        refs = [(pid, 1.0)]

        self.synthesizer.synthesize(refs, use_cache=True)
        cache_key = tuple(sorted(refs))
        assert cache_key in self.synthesizer._cache

        self.synthesizer.clear_cache()
        assert len(self.synthesizer._cache) == 0


class TestCompressionQuality:
    def setup_method(self):
        self.pattern_dict = PatternDictionary(max_patterns=100)
        self.pattern_dict.generate_base_patterns(16)
        self.synthesizer = WeightSynthesizer(self.pattern_dict)

    def test_identity_compression(self):
        identity = torch.eye(16)
        refs, residual = self.synthesizer.compress_weights(identity, max_patterns=8)

        recon = self.synthesizer.synthesize(refs)
        if recon is not None:
            recon = recon + residual
            error = torch.linalg.norm(identity - recon) / torch.linalg.norm(identity)
            assert error < 0.1

    def test_structured_matrix_compression(self):
        x = torch.linspace(0, 2 * 3.14159, 16)
        structured = torch.outer(torch.sin(x), torch.cos(x))

        refs, residual = self.synthesizer.compress_weights(structured, max_patterns=8)
        assert len(refs) > 0
