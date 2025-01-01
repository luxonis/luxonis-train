import logging

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import Tensor

from luxonis_train.utils import Labels, Packet

from .base_visualizer import BaseVisualizer
from .utils import (
    figure_to_torch,
)

logger = logging.getLogger(__name__)
log_disable = False


class EmbeddingsVisualizer(BaseVisualizer[Tensor, Tensor]):
    # supported_tasks: list[TaskType] = [TaskType.LABEL]

    def __init__(
        self,
        **kwargs,
    ):
        """Visualizer for embedding tasks like reID."""
        super().__init__(**kwargs)

    def prepare(
        self, inputs: Packet[Tensor], labels: Labels | None
    ) -> tuple[Tensor, Tensor]:
        embeddings = inputs["features"][0]

        assert (
            labels is not None and "id" in labels
        ), "ID labels are required for metric learning losses"
        ids = labels["id"][0]
        return embeddings, ids

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        embeddings: Tensor,
        ids: Tensor,
        **kwargs,
    ) -> Tensor:
        """Creates a visualization of the embeddings.

        @type label_canvas: Tensor
        @param label_canvas: The canvas to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: The canvas to draw the predictions on.
        @type embeddings: Tensor
        @param embeddings: The embeddings to visualize.
        @type ids: Tensor
        @param ids: The ids to visualize.
        @rtype: Tensor
        @return: An embedding space projection.
        """

        # Embeddings: [B, D], D = e.g. 512
        # ids: [B, 1], corresponding to the embeddings

        # Convert embeddings to numpy array
        embeddings_np = embeddings.detach().cpu().numpy()

        # Perplexity must be less than the number of samples
        perplexity = min(30, embeddings_np.shape[0] - 1)

        # Reduce dimensionality to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_np)

        # Plot the embeddings
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=ids.detach().cpu().numpy(),
            cmap="viridis",
            s=5,
        )

        fig.colorbar(scatter, ax=ax)
        ax.set_title("Embeddings Visualization")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

        # Convert figure to tensor
        image_tensor = figure_to_torch(
            fig, width=label_canvas.shape[3], height=label_canvas.shape[2]
        )

        # Close the figure to free memory
        plt.close(fig)

        # Add fake batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor
