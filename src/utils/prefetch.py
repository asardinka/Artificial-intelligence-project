from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import torch


class CUDAPrefetcher:
    """Асинхронно переносит несколько batch на GPU в отдельном CUDA stream."""

    def __init__(self, loader, device: torch.device, buffer_size: int = 2):
        self.loader = loader
        self.device = device
        self.buffer_size = max(1, int(buffer_size))
        self.stream = torch.cuda.Stream(device=device)
        self.loader_iter = None
        self.buffer: deque[tuple[torch.Tensor, ...]] = deque()
        self.end_reached = False

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        self.loader_iter = iter(self.loader)
        self.buffer.clear()
        self.end_reached = False
        for _ in range(self.buffer_size):
            if not self._preload():
                break
        while self.buffer:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            batch = self.buffer.popleft()
            self._preload()
            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    def _preload(self) -> bool:
        assert self.loader_iter is not None
        if self.end_reached:
            return False
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.end_reached = True
            return False

        with torch.cuda.stream(self.stream):
            moved: list[torch.Tensor] = []
            for tensor in batch:
                moved.append(tensor.to(self.device, non_blocking=True))
            self.buffer.append(tuple(moved))
        return True
