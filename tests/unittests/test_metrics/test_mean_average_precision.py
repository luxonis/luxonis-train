import pytest
import torch
from torch import Size, Tensor

from luxonis_train.attached_modules.metrics.mean_average_precision import (
    MeanAveragePrecisionBBox,
    MeanAveragePrecisionKeypoints,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks
from luxonis_train.utils.dataset_metadata import DatasetMetadata

from .test_utils import convert_bboxes_to_xyxy_and_normalize, normalize_kpts


@pytest.mark.parametrize(
    ("predictions", "targets", "expected"),
    [
        (
            [
                torch.tensor(
                    [
                        [12, 23, 32, 42, 0.9, 0],
                        [18, 27, 37, 48, 0.8, 1],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [0, 1, 15, 25, 35, 45],
                ]
            ),
            torch.tensor(0.3),
        ),
        (
            [
                torch.tensor(
                    [
                        [10, 20, 30, 40, 0.9, 0],
                        [18, 27, 37, 48, 0.8, 1],
                        [50, 60, 70, 80, 1.0, 2],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [0, 1, 15, 25, 35, 45],
                    [0, 2, 50, 60, 70, 80],
                ]
            ),
            torch.tensor(0.766),
        ),
        (
            [
                torch.tensor(
                    [
                        [10, 20, 30, 40, 0.9, 0],
                        [15, 25, 35, 45, 0.8, 1],
                        [50, 60, 70, 80, 0.7, 0],
                        [10, 60, 70, 80, 0.7, 1],
                        [0, 60, 70, 80, 0.7, 1],
                        [5, 60, 70, 80, 0.7, 0],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [0, 1, 15, 25, 35, 45],
                ]
            ),
            torch.tensor(1.0),
        ),
        (
            [
                torch.tensor(
                    [
                        [10, 20, 30, 40, 0.9, 0],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [0, 1, 15, 25, 35, 45],  # FN
                ]
            ),
            torch.tensor(0.5),
        ),
        (
            [
                torch.tensor([[10, 20, 30, 40, 0.9, 0]]),
                torch.tensor(
                    [
                        [15, 25, 35, 45, 0.8, 1],
                        [5, 25, 15, 32, 0.8, 0],
                        [1, 25, 15, 43, 0.8, 0],
                        [5, 25, 15, 32, 0.8, 1],
                        [11, 25, 15, 31, 0.8, 0],
                        [15, 25, 12, 24, 0.8, 0],
                        [11, 25, 14, 33, 0.8, 0],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [1, 1, 13, 25, 31, 45],
                ]
            ),
            torch.tensor(0.75),
        ),
        (
            [torch.tensor([]).reshape(0, 6)],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                ]
            ),
            torch.tensor(0.0),
        ),
        (
            [
                torch.tensor(
                    [
                        [10, 20, 30, 40, 0.9, 0],
                        [12, 22, 29, 39, 0.8, 0],
                        [11, 21, 31, 41, 0.7, 0],
                        [15, 25, 35, 45, 0.95, 1],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [0, 1, 15, 25, 35, 45],
                ]
            ),
            torch.tensor(1.0),
        ),
        (
            [
                torch.tensor(
                    [
                        [10, 20, 30, 40, 0.3, 0],
                        [15, 25, 35, 45, 0.2, 1],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 30, 40],
                    [0, 1, 15, 25, 35, 45],
                ]
            ),
            torch.tensor(1.0),
        ),
        (
            [
                torch.tensor(
                    [
                        [10, 20, 40, 59, 0.9, 0],
                        [12, 22, 37, 54, 0.7, 0],
                        [100, 110, 127, 137, 0.9, 1],
                        [110, 120, 130, 140, 0.2, 1],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 20, 45, 60],
                    [0, 1, 105, 115, 125, 135],
                ]
            ),
            torch.tensor(0.40),
        ),
        (
            [
                torch.tensor(
                    [
                        [12, 12, 48, 48, 0.9, 0],
                        [62, 60, 98, 100, 0.8, 1],
                        [15, 15, 45, 45, 0.75, 0],
                    ]
                )
            ],
            torch.tensor(
                [
                    [0, 0, 10, 10, 50, 50],
                    [0, 1, 60, 60, 100, 100],
                ]
            ),
            torch.tensor(0.80),
        ),
    ],
)
def test_compute_mean_average_precision_bbox(
    predictions: list[Tensor], targets: Tensor, expected: Tensor
):
    class DummyNodeBBox(BaseNode, register=False):
        task = Tasks.BOUNDINGBOX

        def forward(self, _: Tensor) -> Tensor: ...

    image_size = Size([3, 200, 200])
    metric = MeanAveragePrecisionBBox(
        node=DummyNodeBBox(
            n_classes=2,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
            original_in_shape=image_size,
        )
    )

    targets = convert_bboxes_to_xyxy_and_normalize(targets, image_size)

    metric.update(predictions, targets)
    result = metric.compute()[0]
    assert torch.isclose(result, expected, atol=5e-3)


@pytest.mark.parametrize(
    (
        "keypoints",
        "boundingbox",
        "target_keypoints",
        "target_boundingbox",
        "expected",
    ),
    [
        (
            [
                torch.tensor(
                    [
                        [
                            [10, 10, 1],
                            [20, 20, 1],
                            [30, 30, 1],
                            [40, 40, 1],
                        ],
                        [
                            [15, 15, 1],
                            [25, 25, 1],
                            [35, 35, 1],
                            [45, 45, 1],
                        ],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            [10, 10, 1],
                            [20, 20, 1],
                            [30, 30, 1],
                            [40, 40, 1],
                        ],
                        [
                            [15, 15, 1],
                            [25, 25, 1],
                            [35, 35, 1],
                            [45, 45, 1],
                        ],
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        [12, 12, 48, 48, 0.9, 0],
                        [62, 60, 98, 100, 0.8, 1],
                    ]
                ),
                torch.tensor(
                    [
                        [15, 15, 45, 45, 0.75, 0],
                        [65, 65, 95, 95, 0.85, 1],
                    ]
                ),
            ],
            # Target keypoints
            torch.tensor(
                [
                    [0, 10, 10, 1, 20, 20, 1, 30, 30, 1, 40, 40, 1],
                    [0, 19, 12, 2, 20, 30, 2, 30, 40, 2, 40, 50, 2],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 10, 10, 50, 50],
                    [0, 1, 60, 60, 100, 100],
                ]
            ),
            torch.tensor(0.5),
        ),
        (
            [
                torch.tensor(
                    [
                        [
                            [10, 10, 1],
                            [20, 20, 1],
                            [30, 30, 1],
                            [40, 40, 1],
                        ],
                        [
                            [50, 50, 1],
                            [60, 60, 1],
                            [70, 70, 1],
                            [80, 80, 1],
                        ],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            [10, 10, 1],
                            [20, 20, 1],
                            [30, 30, 1],
                            [40, 40, 1],
                        ],
                        [
                            [50, 50, 1],
                            [60, 60, 1],
                            [70, 70, 1],
                            [80, 80, 1],
                        ],
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        [10, 10, 50, 50, 1.0, 0],
                        [60, 60, 100, 100, 1.0, 1],
                    ]
                ),
                torch.tensor(
                    [
                        [10, 10, 50, 50, 1.0, 0],
                        [60, 60, 100, 100, 1.0, 1],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 10, 10, 1, 20, 20, 1, 30, 30, 1, 40, 40, 1],
                    [0, 50, 50, 1, 60, 60, 1, 70, 70, 1, 80, 80, 1],
                    [1, 10, 10, 1, 20, 20, 1, 30, 30, 1, 40, 40, 1],
                    [1, 50, 50, 1, 60, 60, 1, 70, 70, 1, 80, 80, 1],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 10, 10, 50, 50],
                    [0, 1, 60, 60, 100, 100],
                    [1, 0, 10, 10, 50, 50],
                    [1, 1, 60, 60, 100, 100],
                ]
            ),
            torch.tensor(1.0),
        ),
        (
            [
                torch.tensor(
                    [
                        [
                            [110, 110, 2],
                            [120, 120, 2],
                            [130, 130, 2],
                            [140, 140, 2],
                        ],
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        [105, 105, 145, 145, 0.85, 0],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 110, 110, 2, 120, 120, 2, 128, 128, 2, 135, 135, 2],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 100, 100, 145, 135],
                ]
            ),
            torch.tensor(0.30),
        ),
        (
            [
                torch.tensor(
                    [
                        [
                            [10, 10, 1],
                            [20, 20, 1],
                            [31, 30, 1],
                            [39, 40, 1],
                        ],
                        [
                            [56, 55, 1],
                            [66, 66, 1],
                            [74, 75, 1],
                            [86, 85, 1],
                        ],
                    ]
                ),
                torch.tensor(
                    [
                        [
                            [10, 10, 1],
                            [20, 20, 1],
                            [29, 30, 1],
                            [41, 40, 1],
                        ],
                        [
                            [56, 54, 1],
                            [64, 66, 1],
                            [78, 76, 1],
                            [84, 87, 1],
                        ],
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        [14, 14, 48, 48, 0.95, 0],
                        [59, 59, 98, 98, 0.6, 1],
                    ]
                ),
                torch.tensor(
                    [
                        [18, 18, 46, 46, 0.95, 0],
                        [60, 63, 100, 105, 0.85, 1],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 10, 10, 1, 20, 20, 1, 30, 30, 1, 40, 40, 1],
                    [0, 55, 55, 1, 65, 65, 1, 75, 75, 1, 85, 85, 1],
                    [1, 10, 10, 1, 20, 20, 1, 30, 30, 1, 40, 40, 1],
                    [1, 55, 55, 1, 65, 65, 1, 75, 75, 1, 85, 85, 1],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 10, 10, 50, 50],
                    [0, 1, 60, 60, 100, 100],
                    [1, 0, 10, 10, 50, 50],
                    [1, 1, 60, 60, 100, 100],
                ]
            ),
            torch.tensor(0.75),
        ),
        (
            [
                torch.tensor(
                    [
                        [
                            [78, 78, 0.9],
                            [126, 76, 0.9],
                            [126, 126, 0.9],
                            [76, 126, 0.9],
                        ],
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        [50, 50, 150, 150, 0.95, 0],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 75, 75, 2, 125, 75, 2, 125, 125, 2, 75, 125, 2],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 50, 50, 150, 150],
                ]
            ),
            torch.tensor(0.90),
        ),
        (
            [
                torch.tensor(
                    [
                        [
                            [42, 63, 0.8],
                            [87, 59, 0.85],
                            [66, 86, 0.9],
                            [64, 108, 0.75],
                        ],
                    ]
                ),
            ],
            [
                torch.tensor(
                    [
                        [32, 42, 32 + 78, 42 + 88, 0.92, 0],
                    ]
                ),
            ],
            torch.tensor(
                [
                    [0, 40, 60, 2, 90, 60, 2, 65, 85, 2, 65, 110, 2],
                ]
            ),
            torch.tensor(
                [
                    [0, 0, 30, 40, 30 + 80, 40 + 90],
                ]
            ),
            torch.tensor(0.80),
        ),
    ],
)
def test_compute_mean_average_precision_keypoints(
    keypoints: list[Tensor],
    boundingbox: list[Tensor],
    target_keypoints: Tensor,
    target_boundingbox: Tensor,
    expected: Tensor,
):
    class DummyNodeKeypoints(BaseNode, register=False):
        task = Tasks.INSTANCE_KEYPOINTS

        def forward(self, _: Tensor) -> Tensor: ...

    image_size = Size([3, 200, 200])
    sigmas = [0.04, 0.04, 0.04, 0.04]
    area_factor = 0.53
    metric = MeanAveragePrecisionKeypoints(
        sigmas=sigmas,
        area_factor=area_factor,
        node=DummyNodeKeypoints(
            n_classes=2,
            n_keypoints=4,
            dataset_metadata=DatasetMetadata(
                classes={"": {"class1": 0, "class2": 1}}
            ),
            original_in_shape=image_size,
        ),
    )

    target_boundingbox = convert_bboxes_to_xyxy_and_normalize(
        target_boundingbox, image_size
    )
    target_keypoints = normalize_kpts(target_keypoints, image_size)

    metric.update(keypoints, boundingbox, target_keypoints, target_boundingbox)
    result = metric.compute()[0]
    assert torch.isclose(result, expected, atol=5e-3)
