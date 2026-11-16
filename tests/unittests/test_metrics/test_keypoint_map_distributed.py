from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Size, Tensor
from torchmetrics.utilities.distributed import gather_all_tensors

from luxonis_train.attached_modules.metrics.mean_average_precision import (
    MeanAveragePrecisionKeypoints,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class _KeypointNode(BaseNode, register=False):
    task = Tasks.INSTANCE_KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


def _metric(
    class_metrics: bool, **kwargs: Any
) -> MeanAveragePrecisionKeypoints:
    return MeanAveragePrecisionKeypoints(
        sigmas=[0.04] * 4,
        class_metrics=class_metrics,
        node=_KeypointNode(
            n_classes=2,
            n_keypoints=4,
            dataset_metadata=DatasetMetadata(
                classes={"": {"cat": 0, "dog": 1}}
            ),
            original_in_shape=Size([3, 200, 200]),
        ),
        **kwargs,
    )


def _update(metric: MeanAveragePrecisionKeypoints, image: int) -> None:
    # Different counts, false positives, missed objects and empty images.
    classes = [[0, 1], [0], [], [1], [0, 1], []][image]
    pred_classes = [[0, 1, 1], [], [0], [1], [1], []][image]
    points = torch.tensor(
        [
            [30.0, 30.0, 1.0],
            [50.0, 30.0, 1.0],
            [50.0, 50.0, 1.0],
            [30.0, 50.0, 1.0],
        ]
    )
    pred_boxes = torch.tensor(
        [
            [20.0, 20.0, 80.0, 80.0, 0.9 - 0.1 * i, c]
            for i, c in enumerate(pred_classes)
        ]
    ).reshape(-1, 6)
    pred_kpts = points.repeat(len(pred_classes), 1, 1)
    target_boxes = torch.tensor(
        [[0.0, c, 0.1, 0.1, 0.3, 0.3] for c in classes]
    ).reshape(-1, 6)
    target_kpts = points.repeat(len(classes), 1, 1)
    target_kpts[..., :2] /= 200
    target_kpts[..., 2] = 2
    target_kpts = torch.cat(
        (torch.zeros(len(classes), 1), target_kpts.reshape(len(classes), 12)),
        dim=1,
    )
    metric.update([pred_kpts], [pred_boxes], target_kpts, target_boxes)


def _reference(class_metrics: bool) -> tuple[Tensor, dict[str, Tensor]]:
    metric = _metric(class_metrics)
    for image in range(6):
        _update(metric, image)
    return metric.compute()


def _check_epoch(
    metric: MeanAveragePrecisionKeypoints,
    images: range,
    expected: tuple[Tensor, dict[str, Tensor]],
) -> None:
    for image in images:
        _update(metric, image)
    local_shapes = {
        name: [tensor.shape for tensor in getattr(metric, name)]
        for name in metric._defaults
    }
    actual = metric.compute()
    torch.testing.assert_close(actual[0], expected[0])
    assert actual[1].keys() == expected[1].keys()
    for name in expected[1]:
        torch.testing.assert_close(actual[1][name], expected[1][name])
    # Computing restores local states; reset permits a new epoch.
    for name, shapes in local_shapes.items():
        assert [tensor.shape for tensor in getattr(metric, name)] == shapes
    metric.reset()
    assert all(getattr(metric, name) == [] for name in metric._defaults)


def _distributed_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    # Build the reference before distributed initialization so it cannot sync.
    references = [_reference(per_class) for per_class in (False, True)]

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        calls = []

        def gather(tensor: Tensor, group: Any = None) -> list[Tensor]:
            """Record gather invocations while using the real Gloo backend."""
            calls.append(tensor.shape)
            return gather_all_tensors(tensor, group=group)

        for per_class, expected in zip((False, True), references, strict=True):
            metric = _metric(per_class, dist_sync_fn=gather)
            shards = (
                (range(3), range(3, 6)),
                (range(1), range(1, 6)),
                (range(6), range(0)),
            )
            for pair in shards:
                calls.clear()
                _check_epoch(metric, pair[rank], expected)
                assert len(calls) == 2 * len(metric._defaults)

            # A rank with images containing no detections still retains
            # the empty per-image tensors, even if its peer has no images.
            if rank == 0:
                _update(metric, 5)
            metric.sync()
            assert all(
                len(getattr(metric, name)) == 1
                and getattr(metric, name)[0].shape[0] == 0
                for name in metric._defaults
            )
            metric.unsync()
            metric.reset()
            # No updates on any rank must also participate safely.
            metric.sync()
            assert all(
                getattr(metric, name) == [] for name in metric._defaults
            )
            metric.unsync()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo is not available",
)
@pytest.mark.timeout(180)
def test_keypoint_map_distributed_matches_single_process(tmp_path: Path):
    mp.spawn(
        _distributed_worker,
        args=((tmp_path / "rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )
