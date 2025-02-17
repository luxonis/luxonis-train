from collections.abc import Iterator
from copy import deepcopy
from typing import TypeAlias, TypeVar

Graph: TypeAlias = dict[str, list[str]]
"""Graph in a format of a dictionary of predecessors.

Keys are node names, values are inputs to the node (list of node names).
"""


T = TypeVar("T")


def traverse_graph(
    graph: Graph, nodes: dict[str, T]
) -> Iterator[tuple[str, T, list[str], list[str]]]:
    """Traverses the graph in topological order.

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors.
        Keys are node names, values are inputs to the node (list of node
        names).
    @type nodes: dict[str, T]
    @param nodes: Dictionary mapping node names to node objects.
    @rtype: Iterator[tuple[str, T, list[str], list[str]]]
    @return: Iterator of tuples containing node name, node object, node
        dependencies and unprocessed nodes.
    @raises RuntimeError: If the graph is malformed.
    """
    # sort the set to allow reproducibility
    unprocessed_nodes = sorted(set(nodes.keys()))
    processed: set[str] = set()

    graph = deepcopy(graph)
    while unprocessed_nodes:
        unprocessed_nodes_copy = unprocessed_nodes.copy()
        for node_name in unprocessed_nodes_copy:
            node_dependencies = graph[node_name]
            if not node_dependencies or all(
                dependency in processed for dependency in node_dependencies
            ):
                unprocessed_nodes.remove(node_name)
                yield (
                    node_name,
                    nodes[node_name],
                    node_dependencies,
                    unprocessed_nodes.copy(),
                )
                processed.add(node_name)

        if unprocessed_nodes_copy == unprocessed_nodes:
            raise RuntimeError(
                "Malformed graph. "
                "Please check that all nodes are connected in a directed acyclic graph."
            )
