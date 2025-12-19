from __future__ import annotations
import gc
import torch


class MemoryTracker:
    def __init__(self):
        self._snapshots: list[dict] = []

    def snapshot(self, label: str = "") -> dict:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        info = {
            "label": label,
            "cpu_allocated": 0,
            "cuda_allocated": 0,
            "cuda_reserved": 0,
        }

        if torch.cuda.is_available():
            info["cuda_allocated"] = torch.cuda.memory_allocated()
            info["cuda_reserved"] = torch.cuda.memory_reserved()

        self._snapshots.append(info)
        return info

    def report(self) -> str:
        lines = ["Memory Report", "-" * 40]
        for snap in self._snapshots:
            cuda_mb = snap["cuda_allocated"] / (1024 * 1024)
            lines.append(f"{snap['label']}: CUDA={cuda_mb:.2f}MB")
        return "\n".join(lines)

    def reset(self) -> None:
        self._snapshots.clear()

    @staticmethod
    def get_current_usage() -> dict:
        gc.collect()
        result = {"cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
            result["cuda_max_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return result

    @staticmethod
    def clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


def estimate_tensor_size(shape: tuple[int, ...], dtype: torch.dtype = torch.float32) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    bytes_per_element = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
        torch.int32: 4,
        torch.int64: 8,
        torch.int8: 1,
        torch.uint8: 1,
    }.get(dtype, 4)
    return numel * bytes_per_element


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f}PB"
