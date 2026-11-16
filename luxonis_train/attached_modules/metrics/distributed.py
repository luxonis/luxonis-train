"""Distributed gathering helpers for per-image metric states."""

from collections.abc import Callable
from typing import Any

import torch
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

_STATE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
)


def sync_per_image_states(
    metric: Metric,
    dist_sync_fn: Callable = gather_all_tensors,
    process_group: Any = None,
) -> None:
    """Gather per-image vector or matrix lists with two gathers per
    state.

    Opt in only for metrics whose states are all lists of per-image
    tensors that must retain image boundaries. This does not implement
    reductions for scalar states or concatenated sample lists.

    The header records dtype, channel width (-1 for vectors), and per-
    image row counts. The second gather exchanges concatenated rows.
    Empty ranks still participate in both gathers. TorchMetrics caches
    and restores the original local lists around synchronization.

    """
    group = (
        process_group if process_group is not None else metric.process_group
    )
    for name in metric._reductions:
        _sync_per_image_state(metric, name, dist_sync_fn, group)


def _sync_per_image_state(
    metric: Metric, name: str, dist_sync_fn: Callable, group: Any
) -> None:
    """Gather one list state while retaining per-image sizes."""
    values = getattr(metric, name)
    header = [-1, -1]
    if values:
        first = values[0]
        header = [
            _STATE_DTYPES.index(first.dtype),
            first.shape[1] if first.ndim == 2 else -1,
            *(value.shape[0] for value in values),
        ]
    headers = dist_sync_fn(
        torch.tensor(header, device=metric.device, dtype=torch.long),
        group=group,
    )
    metadata = [item.tolist() for item in headers]
    dtype_ids = {item[0] for item in metadata if item[0] != -1}
    if len(dtype_ids) > 1:
        raise ValueError(f"Inconsistent distributed dtypes for '{name}'.")
    dtype = _STATE_DTYPES[next(iter(dtype_ids))] if dtype_ids else metric.dtype
    payload = (
        torch.cat([value.reshape(-1) for value in values]).to(metric.device)
        if values
        else torch.empty(0, device=metric.device, dtype=dtype)
    )
    gathered = dist_sync_fn(payload, group=group)
    images = []
    for data, (_, width, *sizes) in zip(gathered, metadata, strict=True):
        if width != -1:
            data = data.reshape(-1, width)
        images.extend(data.split(sizes))
    setattr(metric, name, images)
