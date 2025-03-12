import math
from typing import Annotated

import torch
from torch import Tensor

from luxonis_train.nodes.heads.ghostfacenet_head import GhostFaceNetHead
from luxonis_train.tasks import Tasks

from .base_metric import BaseMetric, State

# Converted from https://omoindrot.github.io/triplet-loss#offline-and-online-triplet-mining
# to PyTorch from TensorFlow


class ClosestIsPositiveAccuracy(BaseMetric):
    supported_tasks = [Tasks.EMBEDDINGS]
    node: GhostFaceNetHead

    cross_batch_memory: Annotated[
        list[tuple[Tensor, Tensor]], State(default=[], dist_reduce_fx="cat")
    ]
    correct_predictions: Annotated[
        Tensor, State(default=torch.tensor(0), dist_reduce_fx="sum")
    ]
    total_predictions: Annotated[
        Tensor, State(default=torch.tensor(0), dist_reduce_fx="sum")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cross_batch_memory_size = self.node.cross_batch_memory_size

    def update(self, predictions: Tensor, target: Tensor) -> None:
        embeddings, labels = predictions, target

        if self.cross_batch_memory_size is not None:
            self.cross_batch_memory.extend(
                list(zip(embeddings, labels, strict=True))
            )

            if len(self.cross_batch_memory) > self.cross_batch_memory_size:
                self.cross_batch_memory = self.cross_batch_memory[
                    -self.cross_batch_memory_size :
                ]

            if len(self.cross_batch_memory) < self.cross_batch_memory_size:
                return

            embeddings, labels = zip(*self.cross_batch_memory, strict=True)
            embeddings = torch.stack(embeddings)
            labels = torch.stack(labels)

        pairwise_distances = _pairwise_distances(embeddings)
        pairwise_distances.fill_diagonal_(math.inf)

        closest_indices = torch.argmin(pairwise_distances, dim=1)
        closest_labels = labels[closest_indices]

        positive_mask = _get_anchor_positive_triplet_mask(labels)
        n_positives = positive_mask.sum(dim=1)
        has_at_least_one_positive_and_negative = (n_positives > 0) & (
            n_positives < len(labels)
        )

        filtered_labels = labels[has_at_least_one_positive_and_negative]
        filtered_closest_labels = closest_labels[
            has_at_least_one_positive_and_negative
        ]

        correct_predictions = (
            filtered_labels == filtered_closest_labels
        ).sum()

        self.correct_predictions += correct_predictions
        self.total_predictions += len(filtered_labels)

    def compute(self) -> Tensor:
        return self.correct_predictions / self.total_predictions


class MedianDistances(BaseMetric):
    supported_tasks = [Tasks.EMBEDDINGS]
    node: GhostFaceNetHead

    cross_batch_memory: Annotated[
        list[tuple[Tensor, Tensor]], State(default=[], dist_reduce_fx="cat")
    ]
    all_distances: Annotated[
        list[Tensor], State(default=[], dist_reduce_fx="cat")
    ]
    closest_distances: Annotated[
        list[Tensor], State(default=[], dist_reduce_fx="cat")
    ]
    positive_distances: Annotated[
        list[Tensor], State(default=[], dist_reduce_fx="cat")
    ]
    closest_vs_positive_distances: Annotated[
        list[Tensor], State(default=[], dist_reduce_fx="cat")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cross_batch_memory_size = self.node.cross_batch_memory_size

    def update(self, embeddings: Tensor, target: Tensor) -> None:
        if self.cross_batch_memory_size is not None:
            self.cross_batch_memory.extend(
                list(zip(embeddings, target, strict=True))
            )

            if len(self.cross_batch_memory) > self.cross_batch_memory_size:
                self.cross_batch_memory = self.cross_batch_memory[
                    -self.cross_batch_memory_size :
                ]

            if len(self.cross_batch_memory) < self.cross_batch_memory_size:
                return

            embeddings_list, target_list = zip(
                *self.cross_batch_memory, strict=True
            )
            embeddings = torch.stack(embeddings_list)
            target = torch.stack(target_list)

        pairwise_distances = _pairwise_distances(embeddings)
        self.all_distances.append(
            pairwise_distances[
                torch.triu(torch.ones_like(pairwise_distances), diagonal=1)
                == 1
            ].flatten()
        )

        pairwise_distances.fill_diagonal_(math.inf)

        closest_distances, _ = torch.min(pairwise_distances, dim=1)
        self.closest_distances.append(closest_distances)

        positive_mask = _get_anchor_positive_triplet_mask(target).bool()

        only_positive_distances = pairwise_distances.clone()
        only_positive_distances[~positive_mask] = math.inf

        closest_positive_distances, _ = torch.min(
            only_positive_distances, dim=1
        )

        non_inf_mask = closest_positive_distances != math.inf
        difference = closest_positive_distances - closest_distances
        difference = difference[non_inf_mask]

        self.closest_vs_positive_distances.append(difference)
        self.positive_distances.append(
            closest_positive_distances[non_inf_mask]
        )

    def compute(self) -> dict[str, Tensor]:
        if len(self.all_distances) == 0:
            return {
                "MedianDistance": torch.tensor(math.nan),
                "MedianClosestDistance": torch.tensor(math.nan),
                "MedianClosestPositiveDistance": torch.tensor(math.nan),
                "MedianClosestVsClosestPositiveDistance": torch.tensor(
                    math.nan
                ),
            }

        all_distances = torch.cat(self.all_distances)
        closest_distances = torch.cat(self.closest_distances)
        positive_distances = torch.cat(self.positive_distances)
        closest_vs_positive_distances = torch.cat(
            self.closest_vs_positive_distances
        )

        return {
            "MedianDistance": torch.median(all_distances),
            "MedianClosestDistance": torch.median(closest_distances),
            "MedianClosestPositiveDistance": torch.median(positive_distances),
            "MedianClosestVsClosestPositiveDistance": torch.median(
                closest_vs_positive_distances
            ),
        }


def _pairwise_distances(embeddings: Tensor, squared: bool = False) -> Tensor:
    """Compute the 2D matrix of distances between all the embeddings.

    @param embeddings: tensor of shape (batch_size, embed_dim)
    @type embeddings: Tensor
    @param squared: If true, output is the pairwise squared euclidean
        distance matrix. If false, output is the pairwise euclidean
        distance matrix.
    @type squared: bool
    @return: pairwise_distances: tensor of shape (batch_size,
        batch_size)
    @rtype: Tensor
    """
    dot_product = embeddings @ embeddings.T

    square_norm = torch.diag(dot_product)

    distances = (
        square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    )
    distances = torch.max(distances, torch.tensor(0.0))

    if not squared:
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels: Tensor) -> Tensor:
    indices_equal = torch.eye(
        labels.shape[0], dtype=torch.uint8, device=labels.device
    )
    indices_not_equal = ~indices_equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    return indices_not_equal & labels_equal
