from __future__ import annotations
import torch


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = torch.linalg.norm(a_flat)
    norm_b = torch.linalg.norm(b_flat)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return (torch.dot(a_flat, b_flat) / (norm_a * norm_b)).item()


def frobenius_norm(tensor: torch.Tensor) -> float:
    return torch.linalg.norm(tensor.flatten()).item()


def relative_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    diff_norm = frobenius_norm(original - reconstructed)
    orig_norm = frobenius_norm(original)
    if orig_norm < 1e-8:
        return 0.0 if diff_norm < 1e-8 else float("inf")
    return diff_norm / orig_norm


def reconstruct_matrix(
    patterns: list[torch.Tensor],
    scales: list[float],
) -> torch.Tensor:
    if not patterns or not scales:
        raise ValueError("patterns and scales cannot be empty")
    if len(patterns) != len(scales):
        raise ValueError("patterns and scales must have same length")

    result = patterns[0] * scales[0]
    for pattern, scale in zip(patterns[1:], scales[1:]):
        result = result + pattern * scale
    return result


def compute_optimal_scales(
    target: torch.Tensor,
    patterns: list[torch.Tensor],
    regularization: float = 1e-6,
) -> list[float]:
    if not patterns:
        return []

    target_flat = target.flatten()
    pattern_matrix = torch.stack([p.flatten() for p in patterns], dim=1)

    try:
        scales, _ = torch.linalg.lstsq(pattern_matrix, target_flat.unsqueeze(1))
        return scales.squeeze(1).tolist()
    except RuntimeError:
        PtP = pattern_matrix.T @ pattern_matrix
        reg = regularization * torch.eye(PtP.shape[0], device=PtP.device, dtype=PtP.dtype)
        scales = torch.linalg.solve(PtP + reg, pattern_matrix.T @ target_flat)
        return scales.tolist()


def svd_decompose(
    matrix: torch.Tensor,
    rank: int = None,
    energy_threshold: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)

    if rank is None:
        total_energy = torch.sum(S ** 2)
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        rank = torch.searchsorted(cumulative_energy, energy_threshold * total_energy).item() + 1
        rank = max(1, min(rank, len(S)))

    return U[:, :rank], S[:rank], Vh[:rank, :]


def low_rank_approximation(
    matrix: torch.Tensor,
    rank: int = None,
    energy_threshold: float = 0.95,
) -> torch.Tensor:
    U, S, Vh = svd_decompose(matrix, rank, energy_threshold)
    return U @ torch.diag(S) @ Vh
