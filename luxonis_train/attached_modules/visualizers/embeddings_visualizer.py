import logging

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import Tensor

from luxonis_train.enums import Metadata

from .base_visualizer import BaseVisualizer
from .utils import figure_to_torch

logger = logging.getLogger(__name__)
log_disable = False


class EmbeddingsVisualizer(BaseVisualizer[Tensor, Tensor]):
    supported_tasks = [Metadata("id")]

    def __init__(
        self,
        **kwargs,
    ):
        """Visualizer for embedding tasks like reID."""
        super().__init__(**kwargs)

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

        embeddings_np = embeddings.detach().cpu().numpy()

        perplexity = min(30, embeddings_np.shape[0] - 1)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(embeddings_np)

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

        image_tensor = figure_to_torch(
            fig, width=label_canvas.shape[3], height=label_canvas.shape[2]
        )

        plt.close(fig)

        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor
