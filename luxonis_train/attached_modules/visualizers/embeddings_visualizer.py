import colorsys
import logging
from collections.abc import Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn.decomposition import PCA
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
        accumulate_n_batches: int = 2,
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

    def forward(
        self,
        label_canvas: Tensor,
        prediction_canvas: Tensor,
        embeddings: Tensor,
        ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
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
        ids_np = ids.detach().cpu().numpy().astype(int)
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

        z = np.abs(zscore(embeddings_2d))
        mask = (z < 3).all(axis=1)
        embeddings_2d = embeddings_2d[mask]
        ids_np = ids_np[mask]

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

            tensor_image = figure_to_torch(
                fig, width=512, height=512
            ).unsqueeze(0)
            plt.close(fig)
            return tensor_image

        def kde_plot(
            ax: plt.Axes, emb: np.ndarray, labels: np.ndarray
        ) -> None:
            for label in np.unique(labels):
                subset = emb[labels == label]
                color = self._get_color(label)
                sns.kdeplot(
                    x=subset[:, 0],
                    y=subset[:, 1],
                    color=color,
                    alpha=0.9,
                    fill=True,
                    warn_singular=False,
                    ax=ax,
                )

        def scatter_plot(
            ax: plt.Axes, emb: np.ndarray, labels: np.ndarray
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

        kdeplot = plot_to_tensor(embeddings_2d, ids_np, kde_plot)
        scatterplot = plot_to_tensor(embeddings_2d, ids_np, scatter_plot)

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
