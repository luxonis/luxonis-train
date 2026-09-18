"""The helpers behind the Lightning module.

`Nodes` builds the node graph from a config and wraps each node in a
`NodeWrapper` together with its losses, metrics, and visualizers.
`LossAccumulator` keeps the running mean of every loss over an epoch.
The functions sum the losses, build the training strategy, flatten the
metric results, and log the metric artifacts and the visualizations to
the tracker.

"""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import suppress
from functools import cached_property
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NamedTuple,
    TypeVar,
    cast,
)

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from loguru import logger
from luxonis_ml.typing import Params
from luxonis_ml.utils import Registry, traverse_graph
from torch import Size, Tensor, nn
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.attached_modules import BaseLoss, BaseMetric, BaseVisualizer
from luxonis_train.attached_modules.base_attached_module import (
    BaseAttachedModule,
)
from luxonis_train.callbacks import LuxonisModelSummary, TrainingManager
from luxonis_train.callbacks.aimet_callback import AIMETCallback
from luxonis_train.config import AttachedModuleConfig, Config
from luxonis_train.config.config import (
    FinetuningConfig,
    NodeConfig,
    OptimizerConfig,
    SchedulerConfig,
)
from luxonis_train.lightning.freezing import (
    FreezeSchedule,
    resolve_unfreeze_epoch,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.heads.base_head import BaseHead
from luxonis_train.registry import (
    CALLBACKS,
    LOSSES,
    METRICS,
    NODES,
    STRATEGIES,
    VISUALIZERS,
    from_registry,
)
from luxonis_train.strategies import BaseTrainingStrategy
from luxonis_train.strategies.legacy import (
    DEPRECATION_MESSAGE,
    LegacyStrategyAdapter,
)
from luxonis_train.tasks import Metadata
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import DatasetMetadata, LuxonisTrackerPL
from luxonis_train.utils.general import to_shape_packet


class MainMetric(NamedTuple):
    """The metric that selects the best checkpoint.

    Attributes:
        node_name (str): The identifier of the node that holds the
            metric. It is the alias of the node when the config sets
            one, and the class name otherwise.
        metric_name (str): The identifier of the metric. A metric whose
            class name contains ``"ConfusionMatrix"`` gets the
            identifier ``"mcc"``.

    """

    node_name: str
    metric_name: str


class LossAccumulator(defaultdict[str, float]):
    """A running mean of every loss value over one epoch.

    The Lightning module keeps one accumulator for each of the
    ``"train"``, ``"val"``, and ``"test"`` stages. It calls `update`
    after every step and `clear` at the end of the epoch. A key that
    `update` never received reads as ``0.0``.

    Example:
        >>> import torch
        >>> losses = LossAccumulator()
        >>> losses.update({"loss": torch.tensor(2.0)})
        >>> losses.update({"loss": torch.tensor(4.0)})
        >>> losses["loss"]
        3.0
        >>> losses.clear()
        >>> dict(losses)
        {}

    """

    def __init__(self, *args, **kwargs):
        super().__init__(float)
        self._counts = defaultdict(int)

    def update(self, losses: dict[str, Tensor]) -> None:
        """Fold one more value of each loss into its running mean.

        This replaces ``dict.update``. A value does not overwrite the
        stored one. It moves the mean.

        Args:
            losses (``dict[str, Tensor]``): Loss names mapped to
                one-element tensors, as in the second value that
                `compute_losses` returns.

        """
        for key, value in losses.items():
            self[key] = (self[key] * self._counts[key] + value.item()) / (
                self._counts[key] + 1
            )
            self._counts[key] += 1

    def clear(self) -> None:
        """Drop every stored mean and every count."""
        super().clear()
        self._counts.clear()


class NodeWrapper(nn.Module):
    """A node of the graph with its attached modules and its schedule.

    `Nodes` creates one wrapper for each node in the config. The wrapper
    stores every constructor argument under the same name. ``module`` is
    a registered submodule. The losses, the metrics, and the visualizers
    live in plain dictionaries, so the recursive methods of
    `torch.nn.Module` do not reach them.

    """

    def __init__(
        self,
        name: str,
        module: BaseNode,
        losses: dict[str, BaseLoss],
        metrics: dict[str, BaseMetric],
        visualizers: dict[str, BaseVisualizer],
        unfreeze_after: int | None,
        lr_after_unfreeze: float | None,
        finetuning: list[FinetuningConfig],
        inputs: list[str] | None = None,
    ):
        """Initialize the wrapper.

        Args:
            name (str): The identifier of the node in the graph.
            module (BaseNode): The node.
            losses (dict[str, BaseLoss]): The losses attached to the
                node, keyed by their identifier.
            metrics (dict[str, BaseMetric]): The metrics attached to
                the node, keyed by their identifier.
            visualizers (dict[str, BaseVisualizer]): The visualizers
                attached to the node, keyed by their identifier.
            unfreeze_after (int | None): The epoch at which the node
                starts to train. ``None`` when the node is not frozen.
            lr_after_unfreeze (float | None): The base learning rate of
                the node from the unfreeze epoch on. ``None`` keeps the
                rate that the scheduler reached.
            finetuning (list[FinetuningConfig]): The optimizer and
                scheduler overrides of the node.
            inputs (list[str] | None): The names of the nodes and the
                loader sources that feed this node. ``None`` becomes an
                empty list.

        """
        super().__init__()
        self.name = name
        self.module = module
        self.losses = losses
        self.metrics = metrics
        self.visualizers = visualizers
        self.unfreeze_after = unfreeze_after
        self.lr_after_unfreeze = lr_after_unfreeze
        self.finetuning = finetuning
        self.inputs = inputs or []

    @property
    def task_name(self) -> str:
        """The task name of the wrapped node.

        It comes from ``task_name`` in the node config, or from the
        dataset when the dataset holds one task. Otherwise it is an
        empty string.

        """
        return self.module.task_name

    @property
    def formatted_name(self) -> str:
        """The name of the node in the logs and the checkpoint names.

        It is ``"<task_name>-<name>"`` when the node has a task name,
        and ``name`` alone otherwise.

        """
        task_name = self.task_name
        return f"{task_name}-{self.name}" if task_name else self.name

    @override
    def train(self, mode: bool = True) -> "NodeWrapper":
        """Set the training mode of the node and its attached modules.

        `torch.nn.Module.train` reaches only the registered submodules,
        so this override also sets the mode of every loss, metric, and
        visualizer.

        Args:
            mode (bool): ``True`` for training mode, ``False`` for
                evaluation mode.

        Returns:
            NodeWrapper: This wrapper.

        """
        super().train(mode)
        self.module.train(mode)
        for loss in self.losses.values():
            loss.train(mode)
        for metric in self.metrics.values():
            metric.train(mode)
        for visualizer in self.visualizers.values():
            visualizer.train(mode)
        return self


class Nodes(dict[str, NodeWrapper] if TYPE_CHECKING else nn.ModuleDict):
    """The node graph of a model, built from a config.

    A `torch.nn.ModuleDict` that maps each node identifier to its
    `NodeWrapper`. The constructor builds the nodes in topological
    order. It runs each node on zero tensors, so the nodes that follow
    know the shapes of their inputs.

    Attributes:
        graph (dict[str, list[str]]): Each node identifier mapped to
            the identifiers of the nodes that feed it.
        main_metric (MainMetric | None): The metric that selects the
            best checkpoint, or ``None`` when the config has none.
        loader_input_shapes (``dict[str, dict[str, Size]]``): Each node
            identifier mapped to the loader inputs the node reads, as
            input name to shape without the batch dimension. A node fed
            only by other nodes maps to an empty dictionary.
        freeze_schedule (FreezeSchedule): The freeze schedule of the
            nodes with ``freezing.active``.

    """

    def __init__(
        self,
        cfg: Config,
        dataset_metadata: DatasetMetadata,
        input_shapes: dict[str, Size],
    ):
        """Build every node of the config and wrap it.

        Nodes are built in topological order and run once on zero
        tensors to infer the shapes consumed by later nodes. The same
        pass builds each node's losses, metrics, and visualizers.

        Args:
            cfg (Config): The config. ``model.nodes`` lists the nodes,
                ``loader.image_source`` names the image input, and
                ``trainer.epochs`` resolves a fractional or a missing
                ``freezing.unfreeze_after``.
            dataset_metadata (DatasetMetadata): The metadata of the
                dataset. It fixes the task name of a node that sets
                none, and it validates the metadata label types. It
                also reaches every node constructor.
            input_shapes (``dict[str, Size]``): Each loader input name
                mapped to its shape, without the batch dimension.

        Raises:
            RuntimeError: When a node sets no ``task_name`` and the
                dataset holds no task. Also when a node lists an input
                that no node produces, or when the graph has a cycle.
            ValueError: When a head sets no ``task_name`` and the
                dataset holds more than one task. Also when an
                ``input_sources`` entry is not a loader input. Also
                when ``metadata_task_override`` is a string but the
                task does not require exactly one metadata label. Also
                when a metadata label has a type the task does not
                accept. That check reads ``task_name`` from the config
                entry, so it runs only for a node that sets one.

        """
        self._cfg = cfg
        self.graph: dict[str, list[str]] = {}
        self._nodes: dict[str, NodeWrapper] = {}
        self.main_metric = get_main_metric(cfg)

        self.loader_input_shapes = self._get_loader_input_shapes(
            cfg, input_shapes
        )

        dummy_inputs: dict[str, Packet[Tensor]] = {
            input_name: {"features": [torch.zeros(2, *shape)]}
            for shapes in self.loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }

        for node_cfg in cfg.model.nodes:
            self.graph[node_cfg.identifier] = node_cfg.inputs

        for node_name, node_cfg, node_input_names, _ in traverse_graph(
            self.graph, {c.identifier: c for c in cfg.model.nodes}
        ):
            node_dummy_inputs: list[Packet[Tensor]] = []
            node_input_shapes: list[Packet[Size]] = []

            Node = NODES.get(node_cfg.name)
            unfreeze_after, lr_after_unfreeze = self._get_freezing(
                node_cfg, cfg.trainer.epochs
            )
            task_name = self._get_task_name(Node, dataset_metadata, node_cfg)
            self._override_metadata_labels(Node, dataset_metadata, node_cfg)

            if node_cfg.input_sources:
                node_input_names += node_cfg.input_sources

            if not node_cfg.inputs and not node_cfg.input_sources:
                node_input_names += list(input_shapes.keys())

            for node_input_name in node_input_names:
                dummy_input = dummy_inputs[node_input_name]

                node_dummy_inputs.append(dummy_input)

                shape_packet = to_shape_packet(dummy_input)
                node_input_shapes.append(shape_packet)

            node_module = Node(
                **node_cfg.params,
                task_name=task_name,
                remove_on_export=node_cfg.remove_on_export,
                dataset_metadata=dataset_metadata,
                input_shapes=node_input_shapes,
                original_in_shape=input_shapes[cfg.loader.image_source],
                variant=node_cfg.variant,  # type: ignore
            )

            node = NodeWrapper(
                name=node_name,
                module=node_module,
                unfreeze_after=unfreeze_after,
                lr_after_unfreeze=lr_after_unfreeze,
                losses=dict(
                    _init_attached_module(
                        node_module,
                        l_cfg,
                        LOSSES,
                        final_loss_weight=l_cfg.weight,
                    )
                    for l_cfg in node_cfg.losses
                ),
                metrics=dict(
                    _init_attached_module(node_module, m_cfg, METRICS)
                    for m_cfg in node_cfg.metrics
                ),
                visualizers=dict(
                    _init_attached_module(node_module, v_cfg, VISUALIZERS)
                    for v_cfg in node_cfg.visualizers
                ),
                finetuning=node_cfg.finetuning,
                inputs=node_input_names,
            )
            node_outputs = node.module.run(node_dummy_inputs)

            dummy_inputs[node_name] = node_outputs
            self._nodes[node_name] = node

        super().__init__(self._nodes)

        # The schedule snapshots the original trainability state, so
        # build it before any freeze applies.
        self.freeze_schedule = FreezeSchedule.from_nodes(self)

    @cached_property
    def main_metric_reference(self) -> BaseMetric:
        """The metric instance that ``main_metric`` names.

        Raises:
            RuntimeError: When the config defines no main metric.

        """
        if self.main_metric is None:
            raise RuntimeError("Main metric is not defined in the config.")
        node_name, metric_name = self.main_metric
        node = self[node_name]
        return node.metrics[metric_name]

    def _get_task_name(
        self,
        Node: type[BaseNode],
        dataset_metadata: DatasetMetadata,
        node_cfg: NodeConfig,
    ) -> str | None:
        task_name = node_cfg.task_name
        if task_name is None:
            task_names = dataset_metadata.task_names
            if not task_names:
                raise RuntimeError(
                    "Dataset does not contain any labeled images."
                )
            if len(task_names) == 1:
                task_name = next(iter(task_names))
            elif issubclass(Node, BaseHead):
                raise ValueError(
                    f"Dataset contains multiple tasks: {task_names}, "
                    f"but node '{node_cfg.identifier}' does not have the "
                    "`task_name` field specified. "
                    "Please specify the `task_name` parameter "
                    "for each head node. "
                )
        return task_name

    def _override_metadata_labels(
        self,
        Node: type[BaseNode],
        dataset_metadata: DatasetMetadata,
        node_cfg: NodeConfig,
    ) -> None:
        if Node.task is None:
            return
        metadata = {
            label
            for label in Node.task.required_labels
            if isinstance(label, Metadata)
        }
        metadata_override = node_cfg.metadata_task_override
        if metadata_override is not None:
            self._apply_metadata_override(metadata, metadata_override, Node)
        self._validate_metadata_types(metadata, dataset_metadata, node_cfg)

    @staticmethod
    def _apply_metadata_override(
        metadata: set[Metadata],
        metadata_override: str | dict[str, str],
        Node: type[BaseNode],
    ) -> None:
        if isinstance(metadata_override, str):
            if len(metadata) != 1:
                raise ValueError(
                    f"Task '{Node.task}' of node '{Node.__name__}' requires multiple metadata labels: {metadata}, "
                    "so the `metadata_task_override` must be a dictionary."
                )
            metadata_override = {next(iter(metadata)).name: metadata_override}

        for m in metadata:
            m.name = metadata_override.get(m.name, m.name)

    @staticmethod
    def _validate_metadata_types(
        metadata: set[Metadata],
        dataset_metadata: DatasetMetadata,
        node_cfg: NodeConfig,
    ) -> None:
        metadata_types = dataset_metadata.metadata_types

        for m in metadata:
            m_name = f"{node_cfg.task_name}/{m}"
            if m_name not in metadata_types:
                continue
            typ = metadata_types[m_name]
            if not m.check_type(typ):
                raise ValueError(
                    f"Metadata type mismatch for label '{m}' in node '{node_cfg.identifier}'. "
                    f"Expected type '{m.typ}', got '{typ.__name__}'."
                )

    def _get_freezing(
        self, node_cfg: NodeConfig, total_epochs: int
    ) -> tuple[int | None, float | None]:
        unfreeze_after = resolve_unfreeze_epoch(
            node_cfg.freezing, total_epochs
        )
        lr_after_unfreeze = (
            node_cfg.freezing.lr_after_unfreeze
            if node_cfg.freezing.active
            else None
        )
        return unfreeze_after, lr_after_unfreeze

    def _get_loader_input_shapes(
        self, cfg: Config, input_shapes: dict[str, Size]
    ) -> dict[str, dict[str, Size]]:
        loader_input_shapes: dict[str, dict[str, Size]] = {}
        for node in cfg.model.nodes:
            if not node.inputs and not node.input_sources:
                loader_input_shapes[node.identifier] = {
                    k: Size(v) for k, v in input_shapes.items()
                }
            else:
                loader_input_shapes[node.identifier] = {}
                for input_source in node.input_sources:
                    if input_source not in input_shapes:
                        raise ValueError(
                            f"Node '{node.identifier}' requires input source '{input_source}', "
                            "which is not provided by the loader."
                        )

                    loader_input_shapes[node.identifier][input_source] = Size(
                        input_shapes[input_source]
                    )
        return loader_input_shapes

    def formatted_name(self, node_name: str) -> str:
        """Return the log name of a node.

        Args:
            node_name (str): The identifier of the node.

        Returns:
            str: ``"<task_name>-<node_name>"`` when the node has a task
            name, and ``node_name`` alone otherwise.

        """
        return self[node_name].formatted_name

    def traverse(
        self,
    ) -> Iterator[tuple[str, NodeWrapper, list[str], list[str]]]:
        """Walk the graph in topological order.

        The walk yields a node only after every node that feeds it.

        Yields:
            tuple[str, NodeWrapper, list[str], list[str]]: The node
            identifier, its wrapper, the identifiers of the nodes that
            feed it, and the identifiers of the nodes not yet yielded.

        Raises:
            RuntimeError: When the walk makes no progress. A node then
                lists an input that no node of the graph produces, or
                the graph has a cycle.

        """
        yield from traverse_graph(self.graph, self)

    def build_callbacks(self, save_dir: Path) -> list[pl.Callback]:
        """Build the Lightning callbacks of a training run.

        The list holds, in this order:

        - `TrainingManager`, which applies the freeze schedule and
          calls the training strategy after each backward pass.
        - `LuxonisModelSummary` with a depth of 2, as a rich table
          when ``rich_logging`` is true.
        - A ``ModelCheckpoint`` that monitors ``val/loss``, keeps the
          lowest values, and writes to ``save_dir / "min_val_loss"``.
        - `AIMETCallback`, when ``exporter.aimet.active`` is set.
        - A ``ModelCheckpoint`` that monitors
          ``val/metric/<node>/<metric>`` of the main metric, when the
          config defines one. ``<node>`` is the log name from
          `formatted_name`. The checkpoint keeps the highest values
          and writes to ``save_dir / "best_val_metric"``.
        - Every active callback of ``trainer.callbacks``, built from
          the `CALLBACKS` registry. The function logs and skips an
          inactive one.
        - A ``GradientAccumulationScheduler`` for
          ``trainer.accumulate_grad_batches``, when the config sets
          that value and the callbacks above hold no such scheduler.
          When they do, the function logs a warning and ignores the
          config value.

        Both checkpoints keep ``trainer.save_top_k`` files.

        Args:
            save_dir (``Path``): The directory that receives the
                checkpoint subdirectories.

        Returns:
            ``list[pl.Callback]``: The callbacks, in the order above.

        """
        model_name = self._cfg.model.name

        callbacks: list[pl.Callback] = [
            TrainingManager(),
            LuxonisModelSummary(max_depth=2, rich=self._cfg.rich_logging),
            ModelCheckpoint(
                dirpath=save_dir / "min_val_loss",
                filename=f"{model_name}_loss={{val/loss:.4f}}_{{epoch:02d}}",
                monitor="val/loss",
                auto_insert_metric_name=False,
                save_top_k=self._cfg.trainer.save_top_k,
                mode="min",
            ),
        ]

        if self._cfg.exporter.aimet.active:
            callbacks.append(AIMETCallback())

        if self.main_metric is not None:
            node_name, metric_name = self.main_metric
            formatted_node = self.formatted_name(node_name)
            metric_path = f"{formatted_node}/{metric_name}"
            filename_path = metric_path.replace("/", "_")
            callbacks.append(
                ModelCheckpoint(
                    dirpath=save_dir / "best_val_metric",
                    filename=f"{model_name}_{filename_path}="
                    f"{{val/metric/{metric_path}:.4f}}"
                    f"_loss={{val/loss:.4f}}_{{epoch:02d}}",
                    monitor=f"val/metric/{metric_path}",
                    auto_insert_metric_name=False,
                    save_top_k=self._cfg.trainer.save_top_k,
                    mode="max",
                )
            )

        for callback in self._cfg.trainer.callbacks:
            if callback.active:
                callbacks.append(
                    from_registry(CALLBACKS, callback.name, **callback.params)
                )
            else:
                logger.info(f"Callback '{callback.name}' is inactive.")

        if self._cfg.trainer.accumulate_grad_batches is not None:
            if not any(
                isinstance(cb, GradientAccumulationScheduler)
                for cb in callbacks
            ):
                gas = GradientAccumulationScheduler(
                    scheduling={0: self._cfg.trainer.accumulate_grad_batches}
                )
                callbacks.append(gas)
            else:
                logger.warning(
                    "'GradientAccumulationScheduler' is already present "
                    "in the callbacks list. The `accumulate_grad_batches` "
                    "parameter in the config will be ignored."
                )

        return callbacks


def compute_losses(
    cfg: Config,
    losses: dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Sum the losses of every node into one total.

    The total is a plain sum. `BaseLoss.run` already multiplies each
    loss by its ``weight`` from the config. The sub-losses carry no
    weight.

    Args:
        cfg (Config): The config. ``trainer.log_sub_losses`` decides
            whether the sub-losses reach the logged dictionary.
        losses (``dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]]``):
            The losses of one step. The first key is the node
            identifier and the second key is the loss identifier. A
            value is the loss tensor, or a tuple of the loss tensor and
            its sub-losses.
        device (torch.device): The device of the total.

    Returns:
        ``tuple[Tensor, dict[str, Tensor]]``: The total and the losses
        for logging. The total is a tensor of shape ``[1]`` on
        ``device`` that keeps its gradient graph. The dictionary holds
        ``"loss/<node>/<loss>"`` for every loss,
        ``"loss/<node>/<loss>/<sub-loss>"`` for every sub-loss when
        ``trainer.log_sub_losses`` is set, and ``"loss"`` for the total.
        The function detaches every logged tensor and moves it to the
        CPU.

    Example:
        >>> import torch
        >>> from luxonis_train.config import Config
        >>> bce = (torch.tensor(1.5), {"pos": torch.tensor(0.5)})
        >>> total, logged = compute_losses(
        ...     Config(rich_logging=False),
        ...     {"head": {"bce": bce}},
        ...     torch.device("cpu"),
        ... )
        >>> total.tolist()
        [1.5]
        >>> sorted(logged)
        ['loss', 'loss/head/bce', 'loss/head/bce/pos']

    """
    final_loss = torch.zeros(1, device=device)
    all_losses: dict[str, Tensor] = {}
    for node_name, node_losses in losses.items():
        for loss_name, loss_values in node_losses.items():
            if isinstance(loss_values, tuple):
                loss, sublosses = loss_values
            else:
                loss = loss_values
                sublosses = {}

            final_loss += loss
            all_losses[f"loss/{node_name}/{loss_name}"] = loss.detach().cpu()
            if cfg.trainer.log_sub_losses and sublosses:
                for subloss_name, subloss_value in sublosses.items():
                    all_losses[
                        f"loss/{node_name}/{loss_name}/{subloss_name}"
                    ] = subloss_value.detach().cpu()
    all_losses["loss"] = final_loss.detach().cpu()
    return final_loss, all_losses


def build_training_strategy(
    cfg: Config, pl_module: pl.LightningModule
) -> BaseTrainingStrategy | None:
    """Build the training strategy the config names.

    The strategy class comes from the `STRATEGIES` registry, and
    ``trainer.training_strategy.params`` reaches its constructor. A
    strategy supplies the base optimizer and scheduler through
    `BaseTrainingStrategy.get_base_configs`. The function therefore
    logs a warning when ``trainer.optimizer`` or ``trainer.scheduler``
    differs from its default.

    A class without a concrete ``rules`` method predates the rule-based
    API and is deprecated. The function logs a warning. It builds an
    unregistered subclass with stubs for the abstract ``rules`` and
    ``get_base_configs``, so that it can instantiate the class. It
    then mounts the instance through `LegacyStrategyAdapter`. The
    adapter contributes no rules. Its ``get_base_configs`` raises
    ``NotImplementedError`` when the legacy class defines none. The
    training plan then falls back to the optimizer and the scheduler
    of the config.

    Args:
        cfg (Config): The config. ``trainer.training_strategy`` names
            the strategy and holds its parameters.
        pl_module (``pl.LightningModule``): The Lightning module the
            strategy attaches to.

    Returns:
        BaseTrainingStrategy | None: The strategy, or ``None`` when the
        config names none.

    """
    training_strategy = cfg.trainer.training_strategy
    if training_strategy is None:
        return None
    logger.info(f"Using training strategy '{training_strategy.name}'")
    # Warn only about the fields the user changed. Compare the values
    # against the defaults, because `model_fields_set` reports every
    # field as set on a config rebuilt from `model_dump`.
    defaults = {
        "optimizer": OptimizerConfig(),
        "scheduler": SchedulerConfig(),
    }
    overridden = sorted(
        field
        for field, default in defaults.items()
        if getattr(cfg.trainer, field) != default
    )
    if overridden:
        fields = " and ".join(f"`trainer.{field}`" for field in overridden)
        logger.warning(
            "Training strategy is defined. It will override "
            f"the {fields} specified in the config."
        )

    cls = STRATEGIES.get(training_strategy.name)
    rules_attribute = getattr(cls, "rules", None)
    if rules_attribute is None or getattr(
        rules_attribute, "__isabstractmethod__", False
    ):
        # The class predates the rule-based strategy API. Fill the
        # abstract methods with stubs, so Python can instantiate the
        # class. Then mount it through the compatibility adapter.
        logger.warning(DEPRECATION_MESSAGE.format(name=training_strategy.name))

        def no_base_configs(self: Any) -> Any:
            # `resolve_training_plan` falls back to the config's
            # optimizer and scheduler on `NotImplementedError`.
            raise NotImplementedError

        stubs: dict[str, Any] = {}
        for method in getattr(cls, "__abstractmethods__", ()):  # type: ignore[union-attr]
            if method == "rules":
                stubs["rules"] = lambda self: []
            elif method == "get_base_configs":
                stubs["get_base_configs"] = no_base_configs
        # `register=False`: the shim must not replace the original
        # class in the STRATEGIES registry.
        metaclass = cast(Any, type(cls))
        concrete = metaclass(cls.__name__, (cls,), stubs, register=False)
        legacy = cast(Any, concrete)(
            pl_module=pl_module, **training_strategy.params
        )
        return LegacyStrategyAdapter(legacy)

    return from_registry(
        STRATEGIES,
        training_strategy.name,
        **training_strategy.params,
        pl_module=pl_module,
    )


def postprocess_metrics(
    name: str, values: Any, log_sub_metrics: bool = True
) -> dict[str, Tensor]:
    """Flatten the result of `BaseMetric.compute` into named values.

    Args:
        name (str): The identifier of the metric.
        values (``Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]``):
            The computed result. A tensor is the main value. A tuple
            holds the main value and the sub-metrics. A dictionary
            holds only sub-metrics.
        log_sub_metrics (bool): Keep the sub-metrics. When ``False``,
            only the main value remains, and a dictionary result gives
            an empty dictionary.

    Returns:
        ``dict[str, Tensor]``: ``name`` mapped to the main value, plus
        each sub-metric under its own key. A dictionary result gives
        only the sub-metrics.

    Raises:
        ValueError: When ``values`` has none of the three forms.

    Example:
        >>> import torch
        >>> result = (torch.tensor(0.5), {"map_50": torch.tensor(0.75)})
        >>> out = postprocess_metrics("map", result)
        >>> {k: v.item() for k, v in out.items()}
        {'map': 0.5, 'map_50': 0.75}

        >>> out = postprocess_metrics("map", result, log_sub_metrics=False)
        >>> list(out)
        ['map']

        >>> sub = {"map_50": torch.tensor(0.75)}
        >>> postprocess_metrics("map", sub, log_sub_metrics=False)
        {}

    """
    match values:
        case (Tensor(data=value), dict(submetrics)):
            if not log_sub_metrics:
                return {name: value}
            return {name: value} | submetrics
        case Tensor() as value:
            return {name: value}
        case dict(submetrics):
            if not log_sub_metrics:
                return {}
            return submetrics
        case unknown:  # pragma: no cover
            raise ValueError(
                f"Metric '{name}' returned unexpected value of "
                f"type `{type(unknown)}`."
            )


def metric_artifact_image_name(
    mode: Literal["test", "val"],
    formatted_node_name: str,
    metric_name: str,
    artifact_name: str,
) -> str:
    """Build the tracker image name of a metric artifact.

    The name is ``"<mode>/metrics/<node>/<metric>/<artifact>"``. The
    epoch is not part of it. For MLflow, ``log_image`` inserts the step
    as a path segment before the last one, as it does for the
    visualization images. `mlflow_image_key` shows the final MLflow
    path. TensorBoard and Weights and Biases receive the name as it
    is.

    Args:
        mode (``Literal["test", "val"]``): The evaluation stage.
        formatted_node_name (str): The log name of the node, see
            `Nodes.formatted_name`.
        metric_name (str): The identifier of the metric.
        artifact_name (str): The name of the artifact, as
            `BaseMetric.get_artifacts` keys it.

    Returns:
        str: The image name.

    Example:
        >>> metric_artifact_image_name("val", "head", "pr_curve", "curve")
        'val/metrics/head/pr_curve/curve'

    """
    return (
        f"{mode}/metrics/{formatted_node_name}/{metric_name}/{artifact_name}"
    )


def mlflow_image_key(name: str, step: int) -> str:
    """Return the MLflow artifact path of a logged image.

    ``LuxonisTracker.log_image`` splits the caption off the name at the
    last ``/`` and puts the step between the two parts. This function
    builds the same path.
    `LuxonisLightningModule.get_mlflow_logging_keys` uses it to list
    the expected artifacts of a run without a tracker.

    Args:
        name (str): The image name that ``log_image`` receives. It must
            hold at least one ``/``.
        step (int): The step of the image. The Lightning module passes
            the epoch.

    Returns:
        str: ``"<base path>/<step>/<caption>.png"``.

    Raises:
        ValueError: When ``name`` holds no ``/``.

    Example:
        >>> mlflow_image_key("val/metrics/head/pr_curve/curve", 7)
        'val/metrics/head/pr_curve/7/curve.png'

    """
    base_path, caption = name.rsplit("/", 1)
    return f"{base_path}/{step}/{caption}.png"


def log_metric_artifacts(
    tracker: LuxonisTrackerPL,
    metric: BaseMetric,
    computed: Any,
    *,
    mode: Literal["test", "val"],
    formatted_node_name: str,
    metric_name: str,
    current_epoch: int,
) -> None:
    """Render and log the image artifacts of one metric.

    The Lightning module calls it at the end of an evaluation epoch,
    after `BaseMetric.compute` and before the metric resets. The
    module skips the call on the other processes and during the
    sanity check. The function logs every failure and continues, so a
    metric that cannot produce or upload a figure does not stop the
    run. When `BaseMetric.get_artifacts` raises or returns something
    other than a dictionary, the function logs no artifact.

    An artifact must be a tensor of shape ``[C, H, W]``. The function
    logs a warning for any other artifact and skips it. The image goes
    to the tracker as an ``[H, W, C]`` array under the name that
    `metric_artifact_image_name` builds, with ``current_epoch`` as the
    step.

    Args:
        tracker (LuxonisTrackerPL): The tracker that receives the
            images.
        metric (BaseMetric): The metric that produces the artifacts.
        computed (``Tensor | tuple[Tensor, dict[str, Tensor]] | dict[str, Tensor]``):
            The result of `BaseMetric.compute`. The function hands it
            to `BaseMetric.get_artifacts`.
        mode (``Literal["test", "val"]``): The evaluation stage.
        formatted_node_name (str): The log name of the node.
        metric_name (str): The identifier of the metric.
        current_epoch (int): The current epoch, used as the step.

    """
    try:
        artifacts = metric.get_artifacts(computed)
    except Exception:
        logger.exception(
            "Failed to generate artifacts for metric "
            f"'{metric_name}'. Skipping artifact logging."
        )
        return

    if not isinstance(artifacts, dict):
        logger.warning(
            f"Metric '{metric_name}' returned artifacts of type "
            f"'{type(artifacts).__name__}', expected a dict of "
            "images. Skipping artifact logging."
        )
        return

    for artifact_name, artifact in artifacts.items():
        try:
            if not isinstance(artifact, Tensor):
                logger.warning(
                    f"Skipping metric artifact '{artifact_name}' from "
                    f"metric '{metric_name}': expected a tensor of shape "
                    f"[C, H, W], got a '{type(artifact).__name__}'."
                )
                continue
            if artifact.dim() != 3:
                logger.warning(
                    f"Skipping metric artifact '{artifact_name}' from "
                    f"metric '{metric_name}': expected shape [C, H, W], "
                    f"got {tuple(artifact.shape)}."
                )
                continue

            tracker.log_image(
                name=metric_artifact_image_name(
                    mode, formatted_node_name, metric_name, artifact_name
                ),
                img=artifact.detach().cpu().numpy().transpose(1, 2, 0),
                step=current_epoch,
            )
        except Exception:
            logger.exception(
                f"Failed to log metric artifact '{artifact_name}' from "
                f"metric '{metric_name}'. The artifact is dropped for "
                f"epoch {current_epoch}."
            )


T = TypeVar("T", bound=BaseAttachedModule)


def _init_attached_module(
    node: BaseNode,
    cfg: AttachedModuleConfig,
    registry: Registry[type[T]],
    **kwargs,
) -> tuple[str, T]:
    Module = registry.get(cfg.name)
    module_name = cfg.identifier
    params = dict(cfg.params)
    if registry is METRICS:
        params = _translate_predefined_metric_params(
            node, cfg.name, Module, params
        )
    module = Module(**params, node=node, **kwargs)
    if module_name == "ConfusionMatrix":
        module_name = "mcc"
    return module_name, module


def _translate_predefined_metric_params(
    node: BaseNode,
    metric_name: str,
    Module: type[Any],
    params: Params,
) -> Params:
    if "per_class_metrics" not in params:
        return params

    per_class_metrics = params.pop("per_class_metrics")
    if per_class_metrics is None:
        return params

    task = None
    with suppress(RuntimeError):
        task = node.task

    aliases = Module.get_predefined_model_params_aliases(task)
    param_name = aliases.get("per_class_metrics")
    if param_name is None:
        task_name = task.name if task is not None else "unknown"
        logger.warning(
            "Ignoring `per_class_metrics` for metric "
            f"'{metric_name}' on task '{task_name}' because it does not "
            "support a per-class override."
        )
        return params

    params[param_name] = per_class_metrics
    return params


A = TypeVar("A", BaseLoss, BaseMetric, BaseVisualizer)


def log_balanced_class_images(
    tracker: LuxonisTrackerPL,
    nodes: Nodes,
    visualizations: dict[str, dict[str, Tensor]],
    labels: Labels,
    cls_task_keys: list[str],
    class_log_counts: list[int],
    n_logged_images: int,
    max_log_images: int,
    mode: Literal["test", "val"],
    current_epoch: int,
) -> tuple[int, list[int], list[int]]:
    """Log the images of a batch that keep the logged classes balanced.

    The function selects a sample when one of its classes has the
    lowest count in ``class_log_counts`` at that moment. It then adds
    one to the count of every class of the sample. It never selects a
    sample without a present class. Finally, it logs the selected
    samples of every visualization with `log_sequential_images`.

    Args:
        tracker (LuxonisTrackerPL): The tracker that receives the
            images.
        nodes (Nodes): The node graph, used for the log names of the
            nodes.
        visualizations (``dict[str, dict[str, Tensor]]``): The node
            identifier mapped to the visualizer identifier mapped to a
            batch of images of shape ``[B, C, H, W]``. Must not be
            empty, because the function reads the batch size from its
            first entry.
        labels (Labels): The labels of the batch.
        cls_task_keys (list[str]): The label keys that hold multi-label
            classification targets of shape ``[B, n_classes]``. The
            function concatenates the tensors along the class
            dimension, in this order. A class is present when its
            value is above 0.
        class_log_counts (list[int]): How many selected samples held
            each class in this epoch. Its length is the total number
            of classes. The function updates it in place.
        n_logged_images (int): How many images each node logged in
            this epoch before this batch.
        max_log_images (int): The maximum number of images each node
            logs in one epoch.
        mode (``Literal["test", "val"]``): The evaluation stage.
        current_epoch (int): The current epoch, used as the step.

    Returns:
        tuple[int, list[int], list[int]]: The image counter of the last
        node after this batch, as `log_sequential_images` returns it,
        ``class_log_counts`` itself, and the batch indices of the
        selected samples.

    """
    logged_indices = _select_balanced_indices(
        visualizations, labels, cls_task_keys, class_log_counts
    )
    balanced = {
        node_name: {
            viz_name: viz_batch[logged_indices]
            for viz_name, viz_batch in node_visualizations.items()
        }
        for node_name, node_visualizations in visualizations.items()
    }
    node_logged_images = log_sequential_images(
        tracker,
        nodes,
        balanced,
        n_logged_images,
        max_log_images,
        mode,
        current_epoch,
    )
    return node_logged_images, class_log_counts, logged_indices


def log_sequential_images(
    tracker: LuxonisTrackerPL,
    nodes: Nodes,
    visualizations: dict[str, dict[str, Tensor]],
    n_logged_images: int,
    max_log_images: int,
    mode: Literal["test", "val"],
    current_epoch: int,
) -> int:
    """Log the first images of every visualization, in batch order.

    For each node, a counter starts at ``n_logged_images`` and stops
    at ``max_log_images``. Every visualizer of the node shares the
    counter, and the counter names the image. Once the counter reaches
    ``max_log_images``, the rest of the visualizers of the node log
    nothing. The image goes to the tracker under
    ``"<mode>/visualizations/<node>/<visualizer>/<counter>"``, as an
    ``[H, W, C]`` array, with ``current_epoch`` as the step.
    ``<node>`` is the log name from `Nodes.formatted_name`.

    Args:
        tracker (LuxonisTrackerPL): The tracker that receives the
            images.
        nodes (Nodes): The node graph, used for the log names of the
            nodes.
        visualizations (``dict[str, dict[str, Tensor]]``): The node
            identifier mapped to the visualizer identifier mapped to a
            batch of images of shape ``[B, C, H, W]``. Must not be
            empty.
        n_logged_images (int): The counter value each node starts at.
        max_log_images (int): The counter value at which a node stops.
        mode (``Literal["test", "val"]``): The evaluation stage.
        current_epoch (int): The current epoch, used as the step.

    Returns:
        int: The counter of the last node after this batch.

    """
    for node_name, node_visualizations in visualizations.items():
        node_logged_images = n_logged_images
        formatted_node_name = nodes.formatted_name(node_name)
        for viz_name, viz_batch in node_visualizations.items():
            for viz in viz_batch:
                if node_logged_images >= max_log_images:
                    break
                name = (
                    f"{mode}/visualizations/{formatted_node_name}/{viz_name}"
                )
                tracker.log_image(
                    f"{name}/{node_logged_images}",
                    viz.detach().cpu().numpy().transpose(1, 2, 0),
                    step=current_epoch,
                )
                node_logged_images += 1

    return node_logged_images


def compute_visualization_buffer(
    seq_buffer: list[dict[str, dict[str, Tensor]]],
    visualizations: dict[str, dict[str, Tensor]],
    logged_idxs: list[int],
    max_log_images: int,
) -> dict[str, dict[str, Tensor]] | None:
    """Collect the images of a batch that the balanced logger skipped.

    The buffer is a list of batches. Its fill level is the batch
    dimension of the first buffered entry only, not the sum over all
    entries. When that level reaches ``max_log_images``, the function
    collects nothing more. Otherwise the function takes, from every
    visualization, the samples whose indices are not in
    ``logged_idxs``, up to ``max_log_images`` minus the fill level.

    Args:
        seq_buffer (``list[dict[str, dict[str, Tensor]]]``): The batches
            buffered so far. Each entry has the structure of
            ``visualizations``.
        visualizations (``dict[str, dict[str, Tensor]]``): The node
            identifier mapped to the visualizer identifier mapped to a
            batch of images of shape ``[B, C, H, W]``. Must not be
            empty.
        logged_idxs (list[int]): The batch indices that the balanced
            logger already logged.
        max_log_images (int): The number of images to log in one epoch.

    Returns:
        ``dict[str, dict[str, Tensor]] | None``: The skipped samples,
        with the structure of ``visualizations``. ``None`` when the
        first buffered entry is full, or when the balanced logger took
        every sample of the batch.

    Example:
        >>> import torch
        >>> batch = {"head": {"boxes": torch.zeros(4, 3, 8, 8)}}
        >>> extra = compute_visualization_buffer([], batch, [0, 2], 3)
        >>> extra["head"]["boxes"].shape
        torch.Size([2, 3, 8, 8])
        >>> compute_visualization_buffer([extra], batch, [0, 2], 2) is None
        True

    """
    if seq_buffer:
        first_map = seq_buffer[0]
        first_tensor = next(iter(next(iter(first_map.values())).values()))
        buf_count = first_tensor.shape[0]
    else:
        buf_count = 0

    if buf_count >= max_log_images:
        return None

    B = next(iter(next(iter(visualizations.values())).values())).shape[0]
    used = set(logged_idxs)
    free_ix = [i for i in range(B) if i not in used]
    if not free_ix:
        return None

    rem = max_log_images - buf_count
    leftovers: dict[str, dict[str, Tensor]] = {}

    for node_name, viz_map in visualizations.items():
        node_buf: dict[str, Tensor] = {}
        for viz_name, tensor in viz_map.items():
            node_buf[viz_name] = tensor[free_ix][:rem]
        if node_buf:
            leftovers[node_name] = node_buf

    return leftovers or None


def get_model_execution_order(
    model: "lxt.LuxonisLightningModule",
) -> list[str]:
    """List the names of the leaf modules with parameters, in run order.

    The function registers a forward hook on every module that has
    parameters and no child modules. It runs the model on zero tensors
    with a batch size of 2 under ``torch.no_grad``, and removes the
    hooks. The checkpoint stores the list, and
    `LuxonisLightningModule.load_checkpoint` uses it to map the weights
    of an older checkpoint onto a changed module layout.

    **The function leaves the model in evaluation mode.**

    Args:
        model (LuxonisLightningModule): The model. Its
            ``nodes.loader_input_shapes`` gives the input shapes, and
            its ``device`` places the inputs.

    Returns:
        list[str]: The module names, as ``named_modules`` reports them,
        in execution order. A module that runs more than once appears
        once for each run.

    """
    order = []
    handles = []
    model.eval()

    for name, module in model.named_modules():
        if list(module.parameters()) and not list(module.children()):
            handle = module.register_forward_hook(
                lambda mod, inp, out, n=name: order.append(n)
            )
            handles.append(handle)

    with torch.no_grad():
        dummy_inputs = {
            input_name: torch.zeros(2, *shape, device=model.device)
            for shapes in model.nodes.loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }
        model(dummy_inputs)

    for handle in handles:
        handle.remove()

    return order


def get_main_metric(cfg: Config) -> MainMetric | None:
    """Find the metric the config marks with ``is_main_metric``.

    The function scans the nodes and their metrics in config order,
    and the first match wins. The config validation allows one main
    metric at most, and marks the first metric when none is set.

    Args:
        cfg (Config): The config.

    Returns:
        MainMetric | None: The node identifier and the metric
        identifier, or ``None`` when no metric sets ``is_main_metric``.
        A metric whose class name contains ``"ConfusionMatrix"`` gets
        the identifier ``"mcc"``, whatever its alias.

    Example:
        >>> from luxonis_train.config import Config
        >>> node = {"name": "ResNet", "metrics": [{"name": "ConfusionMatrix"}]}
        >>> cfg = Config(rich_logging=False, model={"nodes": [node]})
        >>> get_main_metric(cfg)
        MainMetric(node_name='ResNet', metric_name='mcc')

    """
    for node_cfg in cfg.model.nodes:
        for metric_cfg in node_cfg.metrics:
            if metric_cfg.is_main_metric:
                metric_name = metric_cfg.identifier
                if "ConfusionMatrix" in metric_cfg.name:
                    metric_name = "mcc"
                return MainMetric(node_cfg.identifier, metric_name)
    return None


def check_tensor_device(
    x: Tensor | list[Tensor], device: torch.device
) -> bool:
    """Check whether a tensor, or every tensor of a list, is on a device.

    Args:
        x (``Tensor | list[Tensor]``): The tensor, or a list or tuple
            of tensors.
        device (torch.device): The device to compare with.

    Returns:
        bool: ``True`` when the tensor is on ``device``, or when every
        item of the sequence is a tensor on ``device``. An empty
        sequence gives ``True``.

    Raises:
        TypeError: When ``x`` is neither a tensor nor a list or tuple.

    Example:
        >>> import torch
        >>> cpu = torch.device("cpu")
        >>> check_tensor_device(torch.zeros(1), cpu)
        True
        >>> check_tensor_device([torch.zeros(1), 1.0], cpu)
        False

    """
    if isinstance(x, Tensor):
        return x.device == device
    if isinstance(x, (list | tuple)):
        return all(isinstance(i, Tensor) and i.device == device for i in x)
    raise TypeError(f"Expected Tensor or list[Tensor], got {type(x)!r}")


def _select_balanced_indices(
    visualizations: dict[str, dict[str, Tensor]],
    labels: Labels,
    cls_task_keys: list[str],
    class_log_counts: list[int],
) -> list[int]:
    """Pick the batch indices that keep the logged classes balanced.

    The function updates ``class_log_counts`` in place.

    Args:
        visualizations (``dict[str, dict[str, Tensor]]``): The
            visualizations of the batch. The function reads only the
            batch dimension of the first entry.
        labels (Labels): The labels of the batch.
        cls_task_keys (list[str]): The label keys that hold the
            classification targets, of shape ``[B, n_classes]``.
        class_log_counts (list[int]): How many selected samples held
            each class in this epoch.

    Returns:
        list[int]: The selected batch indices, in ascending order.

    """
    logged_indices = []
    batch_size = next(
        iter(next(iter(visualizations.values())).values())
    ).shape[0]
    cls_tensor = torch.cat([labels[k] for k in cls_task_keys], dim=1)
    present_classes = [
        (cls_tensor[idx] > 0).nonzero(as_tuple=True)[0].tolist()
        for idx in range(batch_size)
    ]
    for idx, classes in enumerate(present_classes):
        if classes:
            min_logged_class = min(classes, key=lambda c: class_log_counts[c])
            if class_log_counts[min_logged_class] == min(class_log_counts):
                logged_indices.append(idx)
                for c in classes:
                    class_log_counts[c] += 1
    return logged_indices
