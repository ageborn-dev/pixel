from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import torch


def save_compressed(
    path: str | Path,
    pattern_refs: dict[str, list[tuple[int, float]]],
    patterns: dict[int, torch.Tensor],
    residuals: dict[str, torch.Tensor] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "pattern_refs": pattern_refs,
        "patterns": {pid: p.cpu() for pid, p in patterns.items()},
        "residuals": {k: v.cpu() for k, v in (residuals or {}).items()},
        "metadata": metadata or {},
    }
    torch.save(data, path)


def load_compressed(
    path: str | Path,
    device: torch.device = None,
) -> tuple[dict, dict, dict, dict]:
    device = device or torch.device("cpu")
    data = torch.load(path, weights_only=False)

    pattern_refs = data.get("pattern_refs", {})
    patterns = {pid: p.to(device) for pid, p in data.get("patterns", {}).items()}
    residuals = {k: v.to(device) for k, v in data.get("residuals", {}).items()}
    metadata = data.get("metadata", {})

    return pattern_refs, patterns, residuals, metadata


def save_config(path: str | Path, config_dict: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def export_onnx_compatible(
    patterns: dict[int, torch.Tensor],
    scales: dict[str, list[float]],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "patterns": {str(k): v.numpy().tolist() for k, v in patterns.items()},
        "scales": scales,
    }
    with open(output_path, "w") as f:
        json.dump(export_data, f)
