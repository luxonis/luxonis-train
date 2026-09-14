"""A loader that yields samples of zeros, for a run without a
dataset.
"""

from collections import defaultdict
from typing import Literal

import torch
from luxonis_ml.typing import check_type
from torch import Size, Tensor
from typing_extensions import override

from luxonis_train.config import Config
from luxonis_train.registry import NODES
from luxonis_train.tasks import Metadata
from luxonis_train.typing import Labels

from .base_loader import BaseLoaderTorch


class DummyLoader(BaseLoaderTorch):
    """A loader that yields samples of zeros.

    `LuxonisModel` uses it in place of the loader of the config when
    ``allow_empty_dataset`` is set and that loader fails to build. The
    samples have the labels that the tasks of the model nodes require,
    so the model can build without data.

    `get_label_shapes` gives a fixed shape to each label type of the
    Luxonis Data Format. A subclass can override it to give the shapes
    of other labels.

    """

    def __init__(
        self,
        cfg: Config,
        view: list[str],
        height: int | None = None,
        width: int | None = None,
        image_source: str = "image",
        color_space: Literal["RGB", "BGR", "GRAY"] = "RGB",
        n_keypoints: int = 3,
        n_classes: int = 1,
        class_names: list[str]
        | dict[str, int]
        | dict[str, dict[str, int]]
        | None = None,
        **kwargs,
    ):
        """Collect the labels of the model and the class names.

        The loader does not store augmentations, so the
        `augmentation_config` property raises ``ValueError``.

        Args:
            cfg (Config): The config. The loader reads
                ``trainer.batch_size`` and the task of each node in
                ``model.nodes``. Each task gives its required labels,
                keyed by the ``task_name`` of the node, or by ``""``
                when the node has no ``task_name``.
            view (list[str]): The splits that form the view.
            height (int | None): The height of the images and the masks.
                With ``None``, `input_shapes` and ``__getitem__`` raise
                ``ValueError``.
            width (int | None): The width of the images and the masks.
                With ``None``, `input_shapes` and ``__getitem__`` raise
                ``ValueError``.
            image_source (str): The input name of the image.
            color_space (``Literal["RGB", "BGR", "GRAY"]``): The color
                space. The image has one channel for ``"GRAY"``, and
                three channels for the other values.
            n_keypoints (int): The number of keypoints of each task.
            n_classes (int): The number of classes of each task when
                ``class_names`` is ``None``.
            class_names (list[str] | dict[str, int] | dict[str, dict[str, int]] | None):
                The classes. A list gives the class ID from the position
                of each name. A list or a ``dict[str, int]`` gives the
                same classes to every task. A ``dict[str, dict[str, int]]``
                maps each task name to its classes. ``None`` gives every
                task the names ``"0"`` to ``str(n_classes - 1)``.
            **kwargs (``Any``): Other loader parameters of the config.
                The loader ignores them.

        """
        super().__init__(
            view=view,
            height=height,
            width=width,
            image_source=image_source,
            color_space=color_space,
        )
        self.n_keypoints = n_keypoints
        self.n_classes = n_classes
        self.batch_size = cfg.trainer.batch_size
        self.labels = self._get_labels(cfg)
        self.n_channels = 1 if color_space == "GRAY" else 3
        self.class_names = self._get_class_names(class_names)

    @staticmethod
    def _get_labels(cfg: Config) -> dict[str, set[str | Metadata]]:
        labels: dict[str, set[str | Metadata]] = defaultdict(set)
        for node in cfg.model.nodes:
            Node = NODES.get(node.name)
            if Node.task is not None:
                for label in Node.task.required_labels:
                    labels[f"{node.task_name or ''}"].add(label)
        return labels

    def _get_class_names(
        self,
        class_names: list[str]
        | dict[str, int]
        | dict[str, dict[str, int]]
        | None,
    ) -> dict[str, dict[str, int]]:
        if isinstance(class_names, list):
            class_names = {name: i for i, name in enumerate(class_names)}
        if check_type(class_names, dict[str, int]):
            class_names = dict.fromkeys(self.labels, class_names)
        if class_names is None:
            class_names = {
                key: {str(i): i for i in range(self.n_classes)}
                for key in self.labels
            }
        return class_names  # type: ignore

    @property
    @override
    def input_shapes(self) -> dict[str, Size]:
        """The shape ``[C, H, W]`` of the image, keyed by
        `image_source`.

        ``C`` is ``1`` for the ``"GRAY"`` color space and ``3`` for the
        other color spaces.

        """
        return {
            self.image_source: Size(
                [
                    self.n_channels,
                    self.height,
                    self.width,
                ]
            )
        }

    @override
    def __len__(self) -> int:
        return self.batch_size * 10

    @override
    def __getitem__(
        self, idx: int
    ) -> tuple[Tensor | dict[str, Tensor], Labels]:
        """Return a sample of zeros.

        Every sample is the same.

        Args:
            idx (int): The index of the sample. The loader ignores it.

        Returns:
            ``tuple[Tensor | dict[str, Tensor], Labels]``: An image of
            zeros of shape ``[C, H, W]``, and a tensor of zeros for each
            required label. A label key is ``"<task_name>/<label>"``.
            `get_label_shapes` gives the shape of each label.

        """
        img = torch.zeros(self.n_channels, self.height, self.width)
        label_shapes = self.get_label_shapes(self.labels)
        labels = {
            f"{task_name}/{task_type}": torch.zeros(
                label_shapes[f"{task_name}/{task_type}"]
            )
            for task_name, task_types in self.labels.items()
            for task_type in task_types
        }
        return img, labels

    @override
    def get_classes(self) -> dict[str, dict[str, int]]:
        """Return the classes that the constructor built.

        Returns:
            dict[str, dict[str, int]]: The class name to class ID
            mapping of each task, keyed by the task name. The
            ``class_names`` argument of the constructor describes the
            content.

        """
        return self.class_names

    @override
    def get_n_keypoints(self) -> dict[str, int] | None:
        """Return ``n_keypoints`` for every task.

        Every task gets the count, also a task without keypoint labels.

        Returns:
            dict[str, int] | None: ``n_keypoints``, keyed by each task
            name of the model.

        """
        return dict.fromkeys(self.labels, self.n_keypoints)

    def get_label_shapes(
        self, labels: dict[str, set[str | Metadata]]
    ) -> dict[str, tuple[int, ...]]:
        """Return the shape of the zero tensor of each label.

        The shape of each label type is:

        - ``boundingbox``: ``[1, 5]``, one box.
        - ``keypoints``: ``[1, 3 * n_keypoints]``, one instance.
        - ``segmentation`` and ``instance_segmentation``:
          ``[1, height, width]``.
        - Any other label, such as ``classification`` or a `Metadata`
          label: ``[2]``.

        A subclass can override the method to give other shapes.

        Args:
            labels (``dict[str, set[str | Metadata]]``): The required
                labels of each task, keyed by the task name.

        Returns:
            ``dict[str, tuple[int, ...]]``: The shape of each label, keyed
            by ``"<task_name>/<label>"``.

        """
        shapes = {}
        for task_name, task_types in labels.items():
            for task_type in task_types:
                name = f"{task_name}/{task_type}"
                match task_type:
                    case "boundingbox":
                        shapes[name] = (1, 5)
                    case "keypoints":
                        shapes[name] = (1, self.n_keypoints * 3)
                    case "segmentation" | "instance_segmentation":
                        shapes[name] = (1, self.height, self.width)
                    case _:
                        shapes[name] = (2,)

        return shapes
