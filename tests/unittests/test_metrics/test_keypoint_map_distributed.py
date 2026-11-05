from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Size, Tensor

from luxonis_train.attached_modules.metrics.mean_average_precision import (
    MeanAveragePrecisionKeypoints,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata


class _KeypointNode(BaseNode, register=False):
    task = Tasks.INSTANCE_KEYPOINTS

    def forward(self, _: Tensor) -> Tensor: ...


def _metric(class_metrics: bool) -> MeanAveragePrecisionKeypoints:
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


def _distributed_worker(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    # Build the reference before distributed initialization so it cannot sync.
    references = []
    for per_class in (False, True):
        reference = _metric(per_class)
        for image in range(6):
            _update(reference, image)
        references.append(reference.compute())

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        for per_class, expected in zip((False, True), references, strict=True):
            metric = _metric(per_class)
            for _ in range(2):
                for image in range(rank * 3, (rank + 1) * 3):
                    _update(metric, image)
                local_shapes = {
                    name: [t.shape for t in getattr(metric, name)]
                    for name in metric._defaults
                }
                actual = metric.compute()
                torch.testing.assert_close(actual[0], expected[0])
                assert actual[1].keys() == expected[1].keys()
                for name in expected[1]:
                    torch.testing.assert_close(
                        actual[1][name], expected[1][name]
                    )
                # Computing restores local states; reset permits a new epoch.
                for name, shapes in local_shapes.items():
                    assert [t.shape for t in getattr(metric, name)] == shapes
                metric.reset()
                assert all(
                    getattr(metric, name) == [] for name in metric._defaults
                )
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
