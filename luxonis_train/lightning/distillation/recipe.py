"""Match student nodes to teacher nodes and choose the distillation
losses.

The recipe runs when training starts. It never changes the config, so a
saved config stays the same when it is loaded again.

"""

from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass

from loguru import logger
from torch import Size

from luxonis_train.config import Config, LossModuleConfig, NodeConfig
from luxonis_train.lightning.utils import Nodes, NodeWrapper
from luxonis_train.nodes.heads import BaseHead
from luxonis_train.registry import NODES


@dataclass(frozen=True)
class NodeMatch:
    """The teacher node that a student node reads.

    Attributes:
        teacher_node: The identifier of the teacher node.
        reason: How the nodes matched, for the log.

    """

    teacher_node: str
    reason: str


@dataclass(frozen=True)
class DistillationEntry:
    """One distillation loss of one student node.

    Attributes:
        student_node: The identifier of the student node.
        teacher_node: The identifier of the teacher node.
        loss: The loss, as a config entry.
        reason: Why the loss is there, for the log.

    """

    student_node: str
    teacher_node: str
    loss: LossModuleConfig
    reason: str


def match_nodes(
    nodes: Nodes, teacher_nodes: Mapping[str, NodeConfig]
) -> dict[str, NodeMatch]:
    """Match the student nodes to the teacher nodes.

    The rules apply in this order. A node that a rule matches keeps
    that match.

    1. The teacher node with the same identifier and the same task.
    2. For a head: the only teacher head with the same task. When more
       than one teacher head has it, the one with the same
       ``task_name``.
    3. The teacher node at the same position in the graph: the inputs
       of a matched node match the inputs of its teacher node in order,
       when both nodes have the same number of inputs.

    Args:
        nodes: The built student nodes.
        teacher_nodes: The node configs of the teacher, keyed by
            identifier, in graph order.

    Returns:
        The matched student nodes, keyed by identifier.

    """
    matches: dict[str, NodeMatch] = {}
    for name, node in nodes.items():
        teacher = teacher_nodes.get(name)
        if teacher is not None and _task_of(teacher) == _task_name(
            node.module.task
        ):
            matches[name] = NodeMatch(name, "same identifier")
    _match_heads_by_task(nodes, teacher_nodes, matches)
    _match_inputs_by_position(nodes, teacher_nodes, matches)
    return matches


def _match_heads_by_task(
    nodes: Nodes,
    teacher_nodes: Mapping[str, NodeConfig],
    matches: dict[str, NodeMatch],
) -> None:
    """Match each free head to the one free teacher head of its task."""
    for name, node in nodes.items():
        if name in matches or node.module.task is None:
            continue
        teacher_head = _head_with_task(
            node,
            teacher_nodes,
            {match.teacher_node for match in matches.values()},
        )
        if teacher_head is not None:
            matches[name] = NodeMatch(teacher_head, "same task")


def _match_inputs_by_position(
    nodes: Nodes,
    teacher_nodes: Mapping[str, NodeConfig],
    matches: dict[str, NodeMatch],
) -> None:
    """Match the inputs of matched nodes in order, from the outputs
    up.
    """
    for name in reversed(list(nodes)):
        if name not in matches:
            continue
        student_inputs = nodes.graph[name]
        teacher_inputs = teacher_nodes[matches[name].teacher_node].inputs
        if len(student_inputs) != len(teacher_inputs):
            continue
        for student, teacher in zip(
            student_inputs, teacher_inputs, strict=True
        ):
            if student in nodes and teacher in teacher_nodes:
                matches.setdefault(
                    student, NodeMatch(teacher, "same graph position")
                )


def resolve_recipe(
    cfg: Config, nodes: Nodes, teacher_nodes: Mapping[str, NodeConfig]
) -> list[DistillationEntry]:
    """List the distillation losses of every student node.

    The ``distillation`` field of each node decides:

    - A list gives these losses. A loss reads its ``teacher_node``, or
      the teacher node that `match_nodes` gives.
    - ``True`` runs the automatic recipe. A matched head whose class
      sets `BaseHead.distillation_loss` gets that loss. A matched node
      that feeds a matched head, with feature maps in its ``features``
      output, gets `ChannelWiseDistillationLoss`.
    - ``False`` gives the node no loss.

    Args:
        cfg: The config of the student.
        nodes: The built student nodes.
        teacher_nodes: The node configs of the teacher, keyed by
            identifier.

    Returns:
        The losses, in graph order.

    Raises:
        ValueError: When an explicit loss has no teacher node, or names
            one that does not exist. Also when no node gets a loss.

    """
    matches = match_nodes(nodes, teacher_nodes)
    configs = {node_cfg.identifier: node_cfg for node_cfg in cfg.model.nodes}
    entries: list[DistillationEntry] = []
    for name in nodes:
        node_cfg = configs[name]
        if isinstance(node_cfg.distillation, list):
            entries += _explicit_entries(node_cfg, matches, teacher_nodes)
        elif node_cfg.distillation:
            entries += _automatic_entries(nodes, name, matches)
    if not entries:
        raise ValueError(_nothing_to_distill(nodes, matches))
    return entries


def levels_read(nodes: Nodes, node_name: str) -> list[int] | None:
    """Find the ``features`` levels of a node that other nodes read.

    Args:
        nodes: The built student nodes.
        node_name: The identifier of the node.

    Returns:
        The indices of the levels, counted from the end, so they also
        select the last levels of a teacher node with more levels.
        ``None`` when the node has no feature maps, or when no node
        reads them.

    """
    if not _has_feature_maps(nodes, node_name):
        return None
    shapes = nodes.output_shapes[node_name]["features"]
    n_levels = len(shapes) if isinstance(shapes, list) else 1
    # One-element sizes stand for the level indices, so `get_attached`
    # applies the attach index of the consumer to them.
    markers = [Size([index]) for index in range(n_levels)]
    read: set[int] = set()
    for consumer, inputs in nodes.graph.items():
        if node_name in inputs:
            attached = nodes[consumer].module.get_attached(markers)
            attached = attached if isinstance(attached, list) else [attached]
            read.update(marker[0] - n_levels for marker in attached)
    return sorted(read) or None


def _explicit_entries(
    node_cfg: NodeConfig,
    matches: Mapping[str, NodeMatch],
    teacher_nodes: Mapping[str, NodeConfig],
) -> Iterator[DistillationEntry]:
    """Read the ``distillation`` list of a node."""
    match = matches.get(node_cfg.identifier)
    for loss_cfg in node_cfg.distillation_losses:
        if loss_cfg.teacher_node is not None:
            teacher_node, reason = loss_cfg.teacher_node, "config"
        elif match is not None:
            teacher_node, reason = match.teacher_node, match.reason
        else:
            raise ValueError(
                f"Distillation loss '{loss_cfg.identifier}' of node "
                f"'{node_cfg.identifier}' has no teacher node: no teacher "
                "node matches the node. Set `teacher_node` to one of "
                f"{list(teacher_nodes)}."
            )
        if teacher_node not in teacher_nodes:
            raise ValueError(
                f"Distillation loss '{loss_cfg.identifier}' of node "
                f"'{node_cfg.identifier}' reads teacher node "
                f"'{teacher_node}', but the teacher has no such node. "
                f"Teacher nodes: {list(teacher_nodes)}."
            )
        yield DistillationEntry(
            node_cfg.identifier, teacher_node, loss_cfg, f"config, {reason}"
        )


def _automatic_entries(
    nodes: Nodes, name: str, matches: Mapping[str, NodeMatch]
) -> Iterator[DistillationEntry]:
    """Apply the automatic recipe to one node."""
    match = matches.get(name)
    module = nodes[name].module
    default = (
        module.distillation_loss if isinstance(module, BaseHead) else None
    )
    if match is None:
        if default is not None:
            logger.info(
                f"Distillation: no teacher node matches head '{name}', "
                "so the head is not distilled."
            )
        return
    if default is not None:
        yield DistillationEntry(
            name,
            match.teacher_node,
            default,
            f"default of the head, {match.reason}",
        )
    elif _feeds_matched_head(nodes, name, matches) and _has_feature_maps(
        nodes, name
    ):
        yield DistillationEntry(
            name,
            match.teacher_node,
            LossModuleConfig(name="ChannelWiseDistillationLoss"),
            f"feeds a matched head, {match.reason}",
        )


def _feeds_matched_head(
    nodes: Nodes, name: str, matches: Mapping[str, NodeMatch]
) -> bool:
    """Tell whether a matched head reads the node."""
    return any(
        name in inputs
        and consumer in matches
        and nodes[consumer].module.task is not None
        for consumer, inputs in nodes.graph.items()
    )


def _head_with_task(
    node: NodeWrapper,
    teacher_nodes: Mapping[str, NodeConfig],
    taken: Collection[str],
) -> str | None:
    """Find the one free teacher head with the task of a head."""
    task = _task_name(node.module.task)
    candidates = [
        name
        for name, cfg in teacher_nodes.items()
        if name not in taken and _task_of(cfg) == task
    ]
    if len(candidates) > 1:
        candidates = [
            name
            for name in candidates
            if teacher_nodes[name].task_name == node.task_name
        ]
    return candidates[0] if len(candidates) == 1 else None


def _task_of(cfg: NodeConfig) -> str | None:
    """Read the task of a node class from the registry."""
    return _task_name(getattr(NODES.get(cfg.name), "task", None))


def _task_name(task: object) -> str | None:
    """Give the name of a task, or ``None`` for a node without one."""
    return getattr(task, "name", None)


def _has_feature_maps(nodes: Nodes, node_name: str) -> bool:
    """Tell whether the ``features`` output holds 4D maps only."""
    shapes = nodes.output_shapes[node_name].get("features")
    if shapes is None:
        return False
    shapes = shapes if isinstance(shapes, list) else [shapes]
    return all(len(shape) == 4 for shape in shapes)


def _nothing_to_distill(nodes: Nodes, matches: Mapping[str, NodeMatch]) -> str:
    """Explain why the recipe gave no node a loss."""
    lines = [
        f"  {name} -> "
        + (
            f"{matches[name].teacher_node} ({matches[name].reason})"
            if name in matches
            else "no teacher node"
        )
        for name in nodes
    ]
    return (
        "`model.teacher` is set, but no node gets a distillation loss. "
        "A matched head gets a loss only when its class sets a default, "
        "and a matched node gets feature distillation only when it feeds "
        "a matched head. The matches:\n"
        + "\n".join(lines)
        + "\nAdd a `distillation` list to a node, or remove "
        "`model.teacher`."
    )
