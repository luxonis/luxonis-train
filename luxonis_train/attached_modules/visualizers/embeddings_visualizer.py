import colorsys
import logging
from collections.abc import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import Tensor

from luxonis_train.tasks import Tasks

from .base_visualizer import BaseVisualizer
from .utils import figure_to_torch

logger = logging.getLogger(__name__)
log_disable = False


class EmbeddingsVisualizer(BaseVisualizer):
    supported_tasks = [Tasks.EMBEDDINGS]

    def __init__(
        self,
        accumulate_n_batches: int = 2,
        z_score_threshold: float = 3,
        **kwargs,
    ):
        """Visualizer for embedding tasks like reID.

        @type accumulate_n_batches: int
        @param accumulate_n_batches: Number of batches to accumulate
            before visualizing.
        """
        super().__init__(**kwargs)
        # self.memory = []
        # self.memory_size = accumulate_n_batches
        self.color_dict = {}
        self.gen = self._distinct_color_generator()
        self.z_score_threshold = z_score_threshold

    def forward(
        self,
        target_canvas: Tensor,
        prediction_canvas: Tensor,
        predictions: Tensor,
        target: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Creates a visualization of the embeddings.

        @type target_canvas: Tensor
        @param target_canvas: The canvas to draw the labels on.
        @type prediction_canvas: Tensor
        @param prediction_canvas: The canvas to draw the predictions on.
        @type embeddings: Tensor
        @param embeddings: The embeddings to visualize.
        @type target: Tensor
        @param target: Ids of the embeddings.
        @rtype: Tensor
        @return: An embedding space projection.
        """

        embeddings_np = predictions.detach().cpu().numpy()
        ids_np = target.detach().cpu().numpy().astype(int)
        # if len(self.memory) < self.memory_size:
        #     self.memory.append((embeddings_np, ids_np))
        #     return None
        #
        # else:
        #     embeddings_np = np.concatenate(
        #         [mem[0] for mem in self.memory], axis=0
        #     )
        #     ids_np = np.concatenate([mem[1] for mem in self.memory], axis=0)
        #     self.memory = []

        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings_np)
        embeddings_2d, ids_np = self._filter_outliers(embeddings_2d, ids_np)

        kdeplot = self.plot_to_tensor(embeddings_2d, ids_np, self.kde_plot)
        scatterplot = self.plot_to_tensor(
            embeddings_2d, ids_np, self.scatter_plot
        )

        return kdeplot, scatterplot

    def _get_color(self, label: int) -> tuple[float, float, float]:
        if label not in self.color_dict:
            self.color_dict[label] = next(self.gen)
        return self.color_dict[label]

    @staticmethod
    def _distinct_color_generator():
        golden_ratio = 0.618033988749895
        hue = 0.0
        while True:
            hue = (hue + golden_ratio) % 1
            saturation = 0.8
            value = 0.95
            yield colorsys.hsv_to_rgb(hue, saturation, value)

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
        palette = {lbl: self._get_color(lbl) for lbl in unique_labels}
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
