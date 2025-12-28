import torch
from torch.utils.data import DataLoader

class HighPerfDataLoader(DataLoader):
    """
    High-performance PyTorch DataLoader subclass with:
    - Multi-worker parallelism
    - Asynchronous prefetching
    - Persistent workers
    - Automatic CPU/GPU-aware pinned memory
    """

    def __init__(
        self,
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    ):
        use_cuda = torch.cuda.is_available()

        # Only include arguments that are valid for the given num_workers
        kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        if num_workers > 0:
            # Only provide prefetch_factor and persistent_workers when workers are used
            kwargs.update(prefetch_factor=4, persistent_workers=True)

        super().__init__(**kwargs)


class PrefetchDataLoader:
    """Simple wrapper that moves batches to a device (cpu/cuda).

    The implementation intentionally keeps the API minimal for tests: it
    accepts any iterable/torch DataLoader and yields batches where any
    tensors (in nested lists/tuples/dicts) have been moved to the target
    device. For CUDA moves we use non_blocking when possible.
    """

    def __init__(self, base_loader, device='cpu'):
        self.base_loader = base_loader
        self.device = torch.device(device)

    def __iter__(self):
        for batch in iter(self.base_loader):
            yield self._move(batch)

    def _move(self, obj):
        if torch.is_tensor(obj):
            # Use non_blocking where it makes sense (e.g., when pin_memory=True)
            return obj.to(self.device, non_blocking=True)
        elif isinstance(obj, (list, tuple)):
            moved = [self._move(x) for x in obj]
            return type(obj)(moved)
        elif isinstance(obj, dict):
            return {k: self._move(v) for k, v in obj.items()}
        else:
            return obj
