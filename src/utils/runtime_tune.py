from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeParams:
    num_workers: int
    prefetch_factor: int
    gpu_prefetch_batches: int
    ram_cache_limit_gb: float


def _available_ram_gb() -> float:
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return float(stat.ullAvailPhys) / (1024**3)
    pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return float(pages * page_size) / (1024**3)


def recommend_runtime_params(
    *,
    device: torch.device,
    batch_size: int,
    image_size: int,
    cpu_cap: int,
    ram_cache_max_gb: float,
) -> RuntimeParams:
    cpu_count = os.cpu_count() or 4
    if device.type != "cuda":
        return RuntimeParams(num_workers=0, prefetch_factor=2, gpu_prefetch_batches=1, ram_cache_limit_gb=1.0)

    workers = max(2, min(cpu_cap, max(2, cpu_count - 2)))
    prefetch = 2 if workers >= 8 else 3
    gpu_prefetch = 4 if batch_size <= 16 else 3

    # Conservative RAM budget: at most 35% of currently available RAM.
    avail_ram = max(1.0, _available_ram_gb())
    ram_limit = max(1.0, min(ram_cache_max_gb, avail_ram * 0.35))

    # If image is large, keep loader pressure moderate to avoid CPU thrash.
    if image_size >= 256:
        workers = min(workers, 6)
        prefetch = min(prefetch, 2)

    return RuntimeParams(
        num_workers=workers,
        prefetch_factor=prefetch,
        gpu_prefetch_batches=gpu_prefetch,
        ram_cache_limit_gb=ram_limit,
    )
