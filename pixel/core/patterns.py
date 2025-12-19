from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from collections import OrderedDict
import torch
import torch.nn.functional as F


class PatternType(str, Enum):
    IDENTITY = "identity"
    DIAGONAL = "diagonal"
    BLOCK = "block"
    FOURIER = "fourier"
    RANDOM = "random"
    LEARNED = "learned"
    SVD = "svd"


@dataclass
class PatternMetadata:
    pattern_type: PatternType
    size: tuple[int, int]
    usage_count: int = 0
    importance_score: float = 0.0
    layer_affinity: list[int] = field(default_factory=list)


class PatternDictionary:
    def __init__(
        self,
        max_patterns: int = 1024,
        cache_size: int = 256,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.max_patterns = max_patterns
        self.cache_size = cache_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self.patterns: dict[int, torch.Tensor] = {}
        self.metadata: dict[int, PatternMetadata] = {}
        self._next_id = 0
        self._cache = OrderedDict()
        self._similarity_threshold = 0.98

    def __len__(self) -> int:
        return len(self.patterns)

    def add(
        self,
        pattern: torch.Tensor,
        pattern_type: PatternType = PatternType.LEARNED,
        check_duplicates: bool = True,
    ) -> int:
        pattern = pattern.to(device=self.device, dtype=self.dtype)
        pattern = self._normalize(pattern)

        if check_duplicates:
            existing_id = self._find_similar(pattern)
            if existing_id is not None:
                self.metadata[existing_id].usage_count += 1
                return existing_id

        if len(self.patterns) >= self.max_patterns:
            self._evict_least_used()

        pattern_id = self._next_id
        self._next_id += 1
        self.patterns[pattern_id] = pattern
        self.metadata[pattern_id] = PatternMetadata(
            pattern_type=pattern_type,
            size=tuple(pattern.shape),
            usage_count=1,
        )
        return pattern_id

    def get(self, pattern_id: int) -> Optional[torch.Tensor]:
        if pattern_id in self._cache:
            self._cache.move_to_end(pattern_id)
            return self._cache[pattern_id]

        if pattern_id not in self.patterns:
            return None

        pattern = self.patterns[pattern_id]
        self.metadata[pattern_id].usage_count += 1

        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[pattern_id] = pattern

        return pattern

    def find_best_matches(
        self,
        target: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.1,
    ) -> list[tuple[int, float]]:
        target = target.to(device=self.device, dtype=self.dtype)
        target_flat = target.flatten()
        target_norm = torch.linalg.norm(target_flat)

        if target_norm < 1e-8:
            return []

        target_normalized = target_flat / target_norm
        matches = []

        for pid, pattern in self.patterns.items():
            pattern_flat = pattern.flatten()
            if pattern_flat.shape != target_flat.shape:
                continue

            pattern_norm = torch.linalg.norm(pattern_flat)
            if pattern_norm < 1e-8:
                continue

            similarity = torch.dot(target_normalized, pattern_flat / pattern_norm).item()
            if abs(similarity) > threshold:
                matches.append((pid, similarity))

        matches.sort(key=lambda x: abs(x[1]), reverse=True)
        return matches[:top_k]

    def prune(self, min_usage: int = None) -> int:
        if min_usage is None:
            min_usage = max(1, len(self.patterns) // 20)

        to_remove = [
            pid for pid, meta in self.metadata.items()
            if meta.usage_count < min_usage
        ]

        for pid in to_remove:
            del self.patterns[pid]
            del self.metadata[pid]
            self._cache.pop(pid, None)

        return len(to_remove)

    def _normalize(self, pattern: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(pattern)
        if norm > 1e-8:
            return pattern / norm
        return pattern

    def _find_similar(self, pattern: torch.Tensor) -> Optional[int]:
        pattern_flat = pattern.flatten()
        for pid, existing in self.patterns.items():
            if existing.shape != pattern.shape:
                continue
            existing_flat = existing.flatten()
            similarity = torch.dot(pattern_flat, existing_flat).item()
            if similarity > self._similarity_threshold:
                return pid
        return None

    def _evict_least_used(self) -> None:
        if not self.metadata:
            return
        min_pid = min(self.metadata.keys(), key=lambda pid: self.metadata[pid].usage_count)
        del self.patterns[min_pid]
        del self.metadata[min_pid]
        self._cache.pop(min_pid, None)

    def generate_base_patterns(self, size: int) -> None:
        identity = torch.eye(size, device=self.device, dtype=self.dtype)
        self.add(identity, PatternType.IDENTITY, check_duplicates=False)

        for k in [-1, 1]:
            diag = torch.diag(torch.ones(size - abs(k), device=self.device, dtype=self.dtype), k)
            if diag.shape[0] == size:
                self.add(diag, PatternType.DIAGONAL, check_duplicates=False)

        x = torch.linspace(0, 2 * torch.pi, size, device=self.device, dtype=self.dtype)
        for freq in [1, 2, 4]:
            sin_pattern = torch.sin(freq * x).unsqueeze(0).expand(size, -1)
            cos_pattern = torch.cos(freq * x).unsqueeze(0).expand(size, -1)
            self.add(sin_pattern, PatternType.FOURIER, check_duplicates=False)
            self.add(cos_pattern, PatternType.FOURIER, check_duplicates=False)

        for block_size in [size // 4, size // 2]:
            if block_size > 0:
                block = torch.zeros(size, size, device=self.device, dtype=self.dtype)
                block[:block_size, :block_size] = 1.0
                self.add(block, PatternType.BLOCK, check_duplicates=False)

    def state_dict(self) -> dict:
        return {
            "patterns": {pid: p.cpu() for pid, p in self.patterns.items()},
            "metadata": self.metadata,
            "next_id": self._next_id,
            "max_patterns": self.max_patterns,
        }

    def load_state_dict(self, state: dict) -> None:
        self.patterns = {pid: p.to(self.device) for pid, p in state["patterns"].items()}
        self.metadata = state["metadata"]
        self._next_id = state["next_id"]
        self.max_patterns = state.get("max_patterns", self.max_patterns)
        self._cache.clear()
