"""Unit tests for pattern dictionary."""
import pytest
import torch

from pixel.core.patterns import PatternDictionary, PatternType


class TestPatternDictionary:
    def setup_method(self):
        self.dict = PatternDictionary(max_patterns=100, cache_size=10)

    def test_add_and_get_pattern(self):
        pattern = torch.randn(4, 4)
        pid = self.dict.add(pattern, PatternType.LEARNED)

        retrieved = self.dict.get(pid)
        assert retrieved is not None
        assert retrieved.shape == pattern.shape

    def test_pattern_normalization(self):
        pattern = torch.randn(4, 4) * 10
        pid = self.dict.add(pattern)

        retrieved = self.dict.get(pid)
        norm = torch.linalg.norm(retrieved)
        assert abs(norm.item() - 1.0) < 1e-5

    def test_duplicate_detection(self):
        pattern = torch.randn(4, 4)
        pid1 = self.dict.add(pattern)
        pid2 = self.dict.add(pattern.clone())

        assert pid1 == pid2
        assert len(self.dict) == 1

    def test_find_best_matches(self):
        p1 = torch.eye(4)
        p2 = torch.randn(4, 4)
        self.dict.add(p1)
        self.dict.add(p2)

        target = torch.eye(4) * 0.9 + torch.randn(4, 4) * 0.1
        matches = self.dict.find_best_matches(target, top_k=2)

        assert len(matches) > 0
        assert matches[0][1] > 0.5

    def test_pruning(self):
        for i in range(10):
            pattern = torch.randn(4, 4)
            self.dict.add(pattern, check_duplicates=False)

        initial_count = len(self.dict)
        removed = self.dict.prune(min_usage=2)

        assert removed > 0
        assert len(self.dict) < initial_count

    def test_generate_base_patterns(self):
        self.dict.generate_base_patterns(8)
        assert len(self.dict) > 5

    def test_eviction_when_full(self):
        small_dict = PatternDictionary(max_patterns=3)
        for i in range(5):
            small_dict.add(torch.randn(4, 4), check_duplicates=False)
        assert len(small_dict) == 3

    def test_state_dict_roundtrip(self):
        self.dict.generate_base_patterns(4)
        pattern = torch.randn(4, 4)
        pid = self.dict.add(pattern)

        state = self.dict.state_dict()
        new_dict = PatternDictionary()
        new_dict.load_state_dict(state)

        assert len(new_dict) == len(self.dict)
        retrieved = new_dict.get(pid)
        assert retrieved is not None


class TestPatternCache:
    def test_cache_hit(self):
        d = PatternDictionary(cache_size=5)
        pid = d.add(torch.randn(4, 4))

        d.get(pid)
        assert pid in d._cache

        d.get(pid)
        assert len(d._cache) == 1

    def test_cache_eviction(self):
        d = PatternDictionary(cache_size=2)
        pids = []
        for _ in range(5):
            pids.append(d.add(torch.randn(4, 4), check_duplicates=False))

        for pid in pids:
            d.get(pid)

        assert len(d._cache) <= 2
