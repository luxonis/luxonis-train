import matplotlib as mpl
import numpy as np
import torch
from torch import Tensor

from luxonis_train.attached_modules.visualizers import EmbeddingsVisualizer
from luxonis_train.nodes import BaseNode
from luxonis_train.tasks import Tasks

# Maximum mean absolute pixel difference (0-255 scale) allowed
# between the generated and reference images.
MAX_MEAN_PIXEL_DIFF = 1.0


class DummyEmbeddingNode(BaseNode, register=False):
    task = Tasks.EMBEDDINGS

    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover
        return x


def _tensor_to_image(tensor: Tensor) -> np.ndarray:
    """Convert a [1, C, H, W] uint8 tensor to a [H, W, C] numpy
    array."""
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def _generate_visualizations() -> tuple[Tensor, Tensor]:
    """Generate the kdeplot and scatterplot tensors
    deterministically."""
    mpl.use("Agg")

    visualizer = EmbeddingsVisualizer(node=DummyEmbeddingNode())

    canvas = torch.zeros(1, 3, 100, 100, dtype=torch.uint8)

    predictions = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0, 10.0],
            [8.0, 9.0, 10.0, 11.0],
            [9.0, 10.0, 11.0, 12.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=torch.float,
    )

    target = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 1], dtype=torch.int)

    kdeplot, scatterplot = visualizer(
        canvas.clone(), canvas.clone(), predictions, target
    )
    return kdeplot, scatterplot


def test_embeddings_visualizer(
    embeddings_visualizer_references: tuple[np.ndarray, np.ndarray],
):
    kde_ref, scatter_ref = embeddings_visualizer_references

    kdeplot, scatterplot = _generate_visualizations()

    kde_generated = _tensor_to_image(kdeplot)
    scatter_generated = _tensor_to_image(scatterplot)

    kde_diff = np.abs(
        kde_generated.astype(np.float32) - kde_ref.astype(np.float32)
    ).mean()
    scatter_diff = np.abs(
        scatter_generated.astype(np.float32) - scatter_ref.astype(np.float32)
    ).mean()

    assert kde_diff < MAX_MEAN_PIXEL_DIFF, (
        f"KDE plot differs from reference by {kde_diff:.2f} mean pixel value "
        f"(threshold: {MAX_MEAN_PIXEL_DIFF})"
    )
    assert scatter_diff < MAX_MEAN_PIXEL_DIFF, (
        f"Scatter plot differs from reference by {scatter_diff:.2f} mean pixel value "
        f"(threshold: {MAX_MEAN_PIXEL_DIFF})"
    )
