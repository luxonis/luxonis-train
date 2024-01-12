import logging
import math
from typing import Generator, TypeVar

from luxonis_ml.data import LuxonisDataset
from pydantic import BaseModel
from torch import Size, Tensor
from torch.utils.data import DataLoader

from luxonis_train.utils.boxutils import anchors_from_dataset
from luxonis_train.utils.types import LabelType, Packet


# TODO: could be moved to luxonis-ml?
# TODO: support multiclass keypoints
class DatasetMetadata:
    """Metadata about the dataset."""

    def __init__(
        self,
        *,
        classes: dict[LabelType, list[str]] | None = None,
        n_classes: int | None = None,
        n_keypoints: int | None = None,
        keypoint_names: list[str] | None = None,
        connectivity: list[tuple[int, int]] | None = None,
        loader: DataLoader | None = None,
    ):
        """An object containing metadata about the dataset. Used to infer the number of
        classes, number of keypoints, I{etc.} instead of passing them as arguments to
        the model.

        @type classes: dict[LabelType, list[str]] | None
        @param classes: Dictionary mapping label types to lists of class names. If not
            provided, will be inferred from the dataset loader.
        @type n_classes: int | None
        @param n_classes: Number of classes for each label type.
        @type n_keypoints: int | None
        @param n_keypoints: Number of keypoints in the dataset.
        @type keypoint_names: list[str] | None
        @param keypoint_names: List of keypoint names.
        @type connectivity: list[tuple[int, int]] | None
        @param connectivity: List of edges in the skeleton graph.
        @type loader: DataLoader | None
        @param loader: Dataset loader.
        """
        if classes is None and n_classes is not None:
            classes = {
                LabelType(lbl): [str(i) for i in range(n_classes)]
                for lbl in LabelType.__members__
            }
        self._classes = classes
        self._keypoint_names = keypoint_names
        self._connectivity = connectivity
        self._n_keypoints = n_keypoints
        if self._n_keypoints is None and self._keypoint_names is not None:
            self._n_keypoints = len(self._keypoint_names)
        self._loader = loader

    @property
    def classes(self) -> dict[LabelType, list[str]]:
        """Dictionary mapping label types to lists of class names.

        @type: dict[LabelType, list[str]]
        @raises ValueError: If classes were not provided during initialization.
        """
        if self._classes is None:
            raise ValueError(
                "Trying to access `classes`, byt they were not"
                "provided during initialization."
            )
        return self._classes

    def n_classes(self, label_type: LabelType | None) -> int:
        """Gets the number of classes for the specified label type.

        @type label_type: L{LabelType} | None
        @param label_type: Label type to get the number of classes for.
        @rtype: int
        @return: Number of classes for the specified label type.
        @raises ValueError: If the dataset loader was not provided during
            initialization.
        @raises ValueError: If the dataset contains different number of classes for
            different label types.
        """
        if label_type is not None:
            if label_type not in self.classes:
                raise ValueError(
                    f"Task type {label_type.name} is not present in the dataset."
                )
            return len(self.classes[label_type])
        n_classes = len(list(self.classes.values())[0])
        for classes in self.classes.values():
            if len(classes) != n_classes:
                raise ValueError(
                    "The dataset contains different number of classes for different tasks."
                )
        return n_classes

    def class_names(self, label_type: LabelType | None) -> list[str]:
        """Gets the class names for the specified label type.

        @type label_type: L{LabelType} | None
        @param label_type: Label type to get the class names for.
        @rtype: list[str]
        @return: List of class names for the specified label type.
        @raises ValueError: If the dataset loader was not provided during
            initialization.
        @raises ValueError: If the dataset contains different class names for different
            label types.
        """
        if label_type is not None:
            if label_type not in self.classes:
                raise ValueError(
                    f"Task type {label_type.name} is not present in the dataset."
                )
            return self.classes[label_type]
        class_names = list(self.classes.values())[0]
        for classes in self.classes.values():
            if classes != class_names:
                raise ValueError(
                    "The dataset contains different class names for different tasks."
                )
        return class_names

    def autogenerate_anchors(self, n_heads: int) -> tuple[list[list[float]], float]:
        """Automatically generates anchors for the provided dataset.

        @type n_heads: int
        @param n_heads: Number of heads to generate anchors for.
        @rtype: tuple[list[list[float]], float]
        @return: List of anchors in [-1,6] format and recall of the anchors.
        @raises ValueError: If the dataset loader was not provided during
            initialization.
        """
        if self.loader is None:
            raise ValueError(
                "Cannot generate anchors without a dataset loader. "
                "Please provide a dataset loader to the constructor "
                "or call `set_loader` method."
            )

        proposed_anchors, recall = anchors_from_dataset(
            self.loader, n_anchors=n_heads * 3
        )
        return proposed_anchors.reshape(-1, 6).tolist(), recall

    def set_loader(self, loader: DataLoader) -> None:
        """Sets the dataset loader.

        @type loader: DataLoader
        @param loader: Dataset loader.
        """
        self.loader = loader

    @classmethod
    def from_dataset(cls, dataset: LuxonisDataset) -> "DatasetMetadata":
        """Creates a L{DatasetMetadata} object from a L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: Dataset to create the metadata from.
        @rtype: DatasetMetadata
        @return: Instance of L{DatasetMetadata} created from the provided dataset.
        """
        _, classes = dataset.get_classes()
        skeletons = dataset.get_skeletons()

        keypoint_names = None
        connectivity = None

        if len(skeletons) == 1:
            name = list(skeletons.keys())[0]
            keypoint_names = skeletons[name]["labels"]
            connectivity = skeletons[name]["edges"]

        elif len(skeletons) > 1:
            raise NotImplementedError(
                "The dataset defines multiclass keypoint detection. "
                "This is not yet supported."
            )

        return cls(
            classes=classes,
            keypoint_names=keypoint_names,
            connectivity=connectivity,
        )


def make_divisible(x: int | float, divisor: int) -> int:
    """Upward revision the value x to make it evenly divisible by the divisor."""
    return math.ceil(x / divisor) * divisor


def infer_upscale_factor(
    in_height: int, orig_height: int, strict: bool = True, warn: bool = True
) -> int:
    """Infer the upscale factor from the input height and original height."""
    num_up = math.log2(orig_height) - math.log2(in_height)
    if num_up.is_integer():
        return int(num_up)
    elif not strict:
        if warn:
            logging.getLogger(__name__).warning(
                f"Upscale factor is not an integer: {num_up}. "
                "Output shape will not be the same as input shape."
            )
        return round(num_up)
    else:
        raise ValueError(
            f"Upscale factor is not an integer: {num_up}. "
            "Output shape will not be the same as input shape."
        )


def get_shape_packet(packet: Packet[Tensor]) -> Packet[Size]:
    shape_packet: Packet[Size] = {}
    for name, value in packet.items():
        shape_packet[name] = [x.shape for x in value]
    return shape_packet


def is_acyclic(graph: dict[str, list[str]]) -> bool:
    """Tests if graph is acyclic.

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors. Keys are node
        names, values are inputs to the node (list of node names).
    @rtype: bool
    @return: True if graph is acyclic, False otherwise.
    """
    graph = graph.copy()

    def dfs(node: str, visited: set[str], recursion_stack: set[str]):
        visited.add(node)
        recursion_stack.add(node)

        for predecessor in graph.get(node, []):
            if predecessor in recursion_stack:
                return True
            if predecessor not in visited:
                if dfs(predecessor, visited, recursion_stack):
                    return True

        recursion_stack.remove(node)
        return False

    visited: set[str] = set()
    recursion_stack: set[str] = set()

    for node in graph.keys():
        if node not in visited:
            if dfs(node, visited, recursion_stack):
                return False

    return True


def validate_packet(data: Packet[Tensor], protocol: type[BaseModel]) -> Packet[Tensor]:
    return protocol(**data).model_dump()


T = TypeVar("T")


# TEST:
def traverse_graph(
    graph: dict[str, list[str]], nodes: dict[str, T]
) -> Generator[tuple[str, T, list[str], set[str]], None, None]:
    """Traverses the graph in topological order.

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors. Keys are node
        names, values are inputs to the node (list of node names).
    @type nodes: dict[str, T]
    @param nodes: Dictionary mapping node names to node objects.
    @rtype: Generator[tuple[str, T, list[str], set[str]], None, None]
    @return: Generator of tuples containing node name, node object, node dependencies
        and unprocessed nodes.
    @raises RuntimeError: If the graph is malformed.
    """
    unprocessed_nodes = set(nodes.keys())
    processed: set[str] = set()

    while unprocessed_nodes:
        unprocessed_nodes_copy = unprocessed_nodes.copy()
        for node_name in unprocessed_nodes_copy:
            node_dependencies = graph[node_name]
            if not node_dependencies or all(
                dependency in processed for dependency in node_dependencies
            ):
                yield node_name, nodes[node_name], node_dependencies, unprocessed_nodes
                processed.add(node_name)
                unprocessed_nodes.remove(node_name)

        if unprocessed_nodes_copy == unprocessed_nodes:
            raise RuntimeError(
                "Malformed graph. "
                "Please check that all nodes are connected in a directed acyclic graph."
            )
