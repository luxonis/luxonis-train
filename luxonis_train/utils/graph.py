from copy import deepcopy
from typing import Generator, TypeVar


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


T = TypeVar("T")


def traverse_graph(
    graph: dict[str, list[str]], nodes: dict[str, T]
) -> Generator[tuple[str, T, list[str], list[str]], None, None]:
    """Traverses the graph in topological order.

    @type graph: dict[str, list[str]]
    @param graph: Graph in a format of a dictionary of predecessors. Keys are node
        names, values are inputs to the node (list of node names).
    @type nodes: dict[str, T]
    @param nodes: Dictionary mapping node names to node objects.
    @rtype: Generator[tuple[str, T, list[str], list[str]], None, None]
    @return: Generator of tuples containing node name, node object, node dependencies
        and unprocessed nodes.
    @raises RuntimeError: If the graph is malformed.
    """
    unprocessed_nodes = sorted(
        set(nodes.keys())
    )  # sort the set to allow reproducibility
    processed: set[str] = set()

    graph = deepcopy(graph)
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
