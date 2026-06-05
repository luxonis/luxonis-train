from collections.abc import Callable

import numpy as np
import seaborn as sns
from loguru import logger
from luxonis_ml.data.utils import ColorMap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor

from luxonis_train.tasks import Tasks

from .base_visualizer import BaseVisualizer
from .utils import figure_to_torch


class EmbeddingsVisualizer(BaseVisualizer):
    """Visualize embedding spaces as two-dimensional plots.

    Metadata:
        - Module type: visualizer
        - Registry name: ``EmbeddingsVisualizer``
        - Task: embeddings
        - Attached node types: None
        - Inputs: prediction and target canvases, ``embeddings`` predictions,
          and metadata ID targets.
        - Outputs: ``(kde_plot, scatter_plot)`` tensors.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Projects embeddings with PCA, filters
          z-score outliers, and renders Seaborn plots as tensors.

    Prediction format:
        - ``predictions`` is a tensor of embedding vectors.

    Target format:
        - ``target`` is a tensor of integer IDs used as plot labels.

    """

    supported_tasks = [Tasks.EMBEDDINGS]

    def __init__(self, z_score_threshold: float = 3, **kwargs):
        """Visualizer for embedding tasks like reID.

        Args:
            z_score_threshold (float): The threshold for filtering out outliers.
            **kwargs (``Any``): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)
        self.colors = ColorMap()
        self.z_score_threshold = z_score_threshold

    def _get_color(self, label: int) -> tuple[float, float, float]:
        r, g, b = self.colors[label]
        return r / 255, g / 255, b / 255

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        predictions: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Create a visualization of the embeddings.

        Args:
            prediction_canvas (``Tensor``): The canvas to draw the predictions on.
            target_canvas (``Tensor``): The canvas to draw the labels on.
            predictions (``Tensor``): Embeddings to visualize.
            target (``Tensor``): IDs of the embeddings.

        Returns:
            ``tuple[Tensor, Tensor]``: KDE and scatter plot projections of the
                embedding space.

        """
        embeddings_np = predictions.detach().cpu().numpy()
        ids_np = target.detach().cpu().numpy().astype(int)

        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings_np)
        if (
            pca.explained_variance_.shape[0] > 1
            and pca.explained_variance_[1] < 1e-12
        ):
            # Stabilize PCA sign when embeddings are effectively 1-D.
            ref = embeddings_np.sum(axis=1)
            if np.dot(embeddings_2d[:, 0], ref) < 0:
                embeddings_2d[:, 0] *= -1
        embeddings_2d, ids_np = self._filter_outliers(embeddings_2d, ids_np)

        kdeplot = self.plot_to_tensor(embeddings_2d, ids_np, self.kde_plot)
        scatterplot = self.plot_to_tensor(
            embeddings_2d, ids_np, self.scatter_plot
        )

        return kdeplot, scatterplot

    def _filter_outliers(
        self, points: np.ndarray, ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.mean(points, axis=0)
        std_dev = np.std(points, axis=0)
        z_scores = (points - mean) / std_dev

        mask = (np.abs(z_scores) < self.z_score_threshold).all(axis=1)
        logger.info(f"Filtered out {len(points) - mask.sum()} outliers")
        return points[mask], ids[mask]

    @staticmethod
    def plot_to_tensor(
        embeddings_2d: np.ndarray,
        ids_np: np.ndarray,
        plot_func: Callable[[plt.Axes, np.ndarray, np.ndarray], None],
    ) -> Tensor:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max())
        ax.set_ylim(embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max())

        plot_func(ax, embeddings_2d, ids_np)
        ax.axis("off")

        tensor_image = figure_to_torch(fig, width=512, height=512).unsqueeze(0)
        plt.close(fig)
        return tensor_image

    def kde_plot(
        self, ax: plt.Axes, emb: np.ndarray, labels: np.ndarray
    ) -> None:
        for label in np.unique(labels):
            subset = emb[labels == label]
            color = self._get_color(label)
            sns.kdeplot(
                x=subset[:, 0],
                y=subset[:, 1],
                color=color,
                alpha=0.9,
                bw_adjust=1.5,
                fill=True,
                warn_singular=False,
                ax=ax,
            )

    def scatter_plot(
        self, ax: plt.Axes, emb: np.ndarray, labels: np.ndarray
    ) -> None:
        unique_labels = np.unique(labels)
        palette = {label: self._get_color(label) for label in unique_labels}
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=labels,
            palette=palette,
            alpha=0.9,
            s=300,
            legend=False,
            ax=ax,
        )
