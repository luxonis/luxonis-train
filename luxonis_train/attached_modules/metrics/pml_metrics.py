import torch
from torch import Tensor

from .base_metric import BaseMetric

# Converted from https://omoindrot.github.io/triplet-loss#offline-and-online-triplet-mining
# to PyTorch from TensorFlow


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    @param embeddings: tensor of shape (batch_size, embed_dim)
    @type embeddings: torch.Tensor
    @param squared: If true, output is the pairwise squared euclidean
        distance matrix. If false, output is the pairwise euclidean
        distance matrix.
    @type squared: bool
    @return: pairwise_distances: tensor of shape (batch_size,
        batch_size)
    @rtype: torch.Tensor
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = (
        square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    )

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.tensor(0.0))

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances


def _get_anchor_positive_triplet_mask(labels):
    indices_equal = torch.eye(
        labels.shape[0], dtype=torch.uint8, device=labels.device
    )
    indices_not_equal = ~indices_equal
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = indices_not_equal & labels_equal
    return mask


class ClosestIsPositiveAccuracy(BaseMetric):
    def __init__(self, cross_batch_memory_size=0, **kwargs):
        super().__init__(**kwargs)
        self.cross_batch_memory_size = cross_batch_memory_size
        self.add_state("cross_batch_memory", default=[], dist_reduce_fx="cat")
        self.add_state(
            "correct_predictions",
            default=torch.tensor(0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total_predictions", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def prepare(self, inputs, labels):
        embeddings = inputs["features"][0]

        assert (
            labels is not None and "id" in labels
        ), "ID labels are required for metric learning losses"
        IDs = labels["id"][0][:, 0]
        return embeddings, IDs

    def update(self, inputs: Tensor, target: Tensor):
        embeddings, labels = inputs, target

        if self.cross_batch_memory_size > 0:
            # Append embedding and labels to the memory
            self.cross_batch_memory.extend(list(zip(embeddings, labels)))

            # If the memory is full, remove the oldest elements
            if len(self.cross_batch_memory) > self.cross_batch_memory_size:
                self.cross_batch_memory = self.cross_batch_memory[
                    -self.cross_batch_memory_size :
                ]

            # If the memory is not full, return
            if len(self.cross_batch_memory) < self.cross_batch_memory_size:
                return

            # Get the embeddings and labels from the memory
            embeddings, labels = zip(*self.cross_batch_memory)
            embeddings = torch.stack(embeddings)
            labels = torch.stack(labels)

        # print(f"Calculating accuracy for {len(embeddings)} embeddings")

        # Get the pairwise distances between all embeddings
        pairwise_distances = _pairwise_distances(embeddings)

        # Set diagonal to infinity so that the closest embedding is not the same embedding
        pairwise_distances.fill_diagonal_(float("inf"))

        # Find the closest embedding for each query embedding
        closest_indices = torch.argmin(pairwise_distances, dim=1)

        # Get the labels of the closest embeddings
        closest_labels = labels[closest_indices]

        # Filter out embeddings that don't have both positive and negative examples
        positive_mask = _get_anchor_positive_triplet_mask(labels)
        num_positives = positive_mask.sum(dim=1)
        has_at_least_one_positive_and_negative = (num_positives > 0) & (
            num_positives < len(labels)
        )

        # Filter embeddings, labels, and closest indices based on valid indices
        filtered_labels = labels[has_at_least_one_positive_and_negative]
        filtered_closest_labels = closest_labels[
            has_at_least_one_positive_and_negative
        ]

        # Calculate the number of correct predictions where the closest is positive
        correct_predictions = (
            filtered_labels == filtered_closest_labels
        ).sum()

        # Update the metric state
        self.correct_predictions += correct_predictions
        self.total_predictions += len(filtered_labels)

    def compute(self):
        return self.correct_predictions / self.total_predictions


class MedianDistances(BaseMetric):
    def __init__(self, cross_batch_memory_size=0, **kwargs):
        super().__init__(**kwargs)
        self.cross_batch_memory_size = cross_batch_memory_size
        self.add_state("cross_batch_memory", default=[], dist_reduce_fx="cat")
        self.add_state("all_distances", default=[], dist_reduce_fx="cat")
        self.add_state("closest_distances", default=[], dist_reduce_fx="cat")
        self.add_state("positive_distances", default=[], dist_reduce_fx="cat")
        self.add_state(
            "closest_vs_positive_distances", default=[], dist_reduce_fx="cat"
        )

    def prepare(self, inputs, labels):
        embeddings = inputs["features"][0]

        assert (
            labels is not None and "id" in labels
        ), "ID labels are required for metric learning losses"
        IDs = labels["id"][0][:, 0]
        return embeddings, IDs

    def update(self, inputs: Tensor, target: Tensor):
        embeddings, labels = inputs, target

        if self.cross_batch_memory_size > 0:
            # Append embedding and labels to the memory
            self.cross_batch_memory.extend(list(zip(embeddings, labels)))

            # If the memory is full, remove the oldest elements
            if len(self.cross_batch_memory) > self.cross_batch_memory_size:
                self.cross_batch_memory = self.cross_batch_memory[
                    -self.cross_batch_memory_size :
                ]

            # If the memory is not full, return
            if len(self.cross_batch_memory) < self.cross_batch_memory_size:
                return

            # Get the embeddings and labels from the memory
            embeddings, labels = zip(*self.cross_batch_memory)
            embeddings = torch.stack(embeddings)
            labels = torch.stack(labels)

        # Get the pairwise distances between all embeddings
        pairwise_distances = _pairwise_distances(embeddings)
        # Append only upper triangular part of the matrix
        self.all_distances.append(
            pairwise_distances[
                torch.triu(torch.ones_like(pairwise_distances), diagonal=1)
                == 1
            ].flatten()
        )

        # Set diagonal to infinity so that the closest embedding is not the same embedding
        pairwise_distances.fill_diagonal_(float("inf"))

        # Get the closest distance for each query embedding
        closest_distances, _ = torch.min(pairwise_distances, dim=1)
        self.closest_distances.append(closest_distances)

        # Get the positive mask and convert it to boolean
        positive_mask = _get_anchor_positive_triplet_mask(labels).bool()

        only_positive_distances = pairwise_distances.clone()
        only_positive_distances[~positive_mask] = float("inf")

        closest_positive_distances, _ = torch.min(
            only_positive_distances, dim=1
        )

        non_inf_mask = closest_positive_distances != float("inf")
        difference = closest_positive_distances - closest_distances
        difference = difference[non_inf_mask]

        # Update the metric state
        self.closest_vs_positive_distances.append(difference)
        self.positive_distances.append(
            closest_positive_distances[non_inf_mask]
        )

    def compute(self):
        if len(self.all_distances) == 0:
            # Return NaN tensor if no distances were calculated
            return {
                "MedianDistance": torch.tensor(float("nan")),
                "MedianClosestDistance": torch.tensor(float("nan")),
                "MedianClosestPositiveDistance": torch.tensor(float("nan")),
                "MedianClosestVsClosestPositiveDistance": torch.tensor(
                    float("nan")
                ),
            }

        all_distances = torch.cat(self.all_distances)
        closest_distances = torch.cat(self.closest_distances)
        positive_distances = torch.cat(self.positive_distances)
        closest_vs_positive_distances = torch.cat(
            self.closest_vs_positive_distances
        )

        # Return medians
        return {
            "MedianDistance": torch.median(all_distances),
            "MedianClosestDistance": torch.median(closest_distances),
            "MedianClosestPositiveDistance": torch.median(positive_distances),
            "MedianClosestVsClosestPositiveDistance": torch.median(
                closest_vs_positive_distances
            ),
        }
