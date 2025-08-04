from collections import defaultdict  # noqa: INP001
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeVar

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from loguru import logger
from luxonis_ml.typing import ConfigItem, Kwargs, check_type
from luxonis_ml.utils import traverse_graph
from luxonis_ml.utils.registry import Registry
from torch import Size, Tensor, nn
from torch.optim.lr_scheduler import LRScheduler, SequentialLR
from torch.optim.optimizer import Optimizer

import luxonis_train as lxt
from luxonis_train.attached_modules import BaseLoss, BaseMetric, BaseVisualizer
from luxonis_train.callbacks import LuxonisModelSummary, TrainingManager
from luxonis_train.config import AttachedModuleConfig, Config
from luxonis_train.nodes import BaseHead, BaseNode
from luxonis_train.registry import (
    CALLBACKS,
    LOSSES,
    METRICS,
    NODES,
    OPTIMIZERS,
    SCHEDULERS,
    STRATEGIES,
    VISUALIZERS,
    from_registry,
)
from luxonis_train.strategies import BaseTrainingStrategy
from luxonis_train.tasks import Metadata
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import DatasetMetadata, LuxonisTrackerPL
from luxonis_train.utils.general import to_shape_packet

A = TypeVar("A", BaseLoss, BaseMetric, BaseVisualizer)

AttachedModulesDict: TypeAlias = dict[str, dict[str, A]]


class LossAccumulator(defaultdict[str, float]):
    def __init__(self):
        super().__init__(float)
        self.counts = defaultdict(int)

    def update(self, losses: dict[str, Tensor]) -> None:
        for key, value in losses.items():
            self[key] = (self[key] * self.counts[key] + value.item()) / (
                self.counts[key] + 1
            )
            self.counts[key] += 1

    def clear(self) -> None:
        super().clear()
        self.counts.clear()


class Nodes(dict[str, BaseNode] if TYPE_CHECKING else nn.ModuleDict):
    def __init__(
        self,
        node_kwargs: dict[str, tuple[type[BaseNode], Kwargs]],
        loader_input_shapes: dict[str, dict[str, Size]],
        inputs: dict[str, list[str]],
        graph: dict[str, list[str]],
        task_names: dict[str, str],
        frozen_nodes: dict[str, tuple[int, float | None]],
    ):
        self.graph = graph
        self.task_names = task_names
        self.inputs = defaultdict(list, inputs)
        self.unfreeze_after = frozen_nodes
        self.input_shapes = loader_input_shapes

        initiated_nodes: dict[str, BaseNode] = {}

        dummy_inputs: dict[str, Packet[Tensor]] = {
            input_name: {"features": [torch.zeros(2, *shape)]}
            for shapes in loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }

        for node_name, (Node, kwargs), node_input_names, _ in traverse_graph(
            graph, node_kwargs
        ):
            node_dummy_inputs: list[Packet[Tensor]] = []
            node_input_shapes: list[Packet[Size]] = []

            node_input_names += self.inputs[node_name]
            for node_input_name in node_input_names:
                dummy_input = dummy_inputs[node_input_name]

                node_dummy_inputs.append(dummy_input)

                shape_packet = to_shape_packet(dummy_input)
                node_input_shapes.append(shape_packet)

            node = Node(input_shapes=node_input_shapes, **kwargs)

            if isinstance(node, BaseHead):
                try:
                    node.get_custom_head_config()
                except NotImplementedError:
                    logger.warning(
                        f"Head {node_name} does not implement "
                        "`get_custom_head_config` method. "
                        "Archivation of this head will fail."
                    )
            node_outputs = node.run(node_dummy_inputs)

            dummy_inputs[node_name] = node_outputs
            initiated_nodes[node_name] = node

        super().__init__(initiated_nodes)

    @property
    def any_frozen(self) -> bool:
        return bool(self.unfreeze_after)

    def formatted_name(self, node_name: str) -> str:
        task_name = self.task_names[node_name]
        return f"{task_name}-{node_name}" if task_name else node_name

    def is_frozen(self, node_name: str) -> bool:
        return self.unfreeze_after.get(node_name, 0) == 0

    def frozen_nodes(
        self,
    ) -> Iterator[tuple[str, BaseNode, int, float | None]]:
        for node_name, (
            unfreeze_after,
            lr_after_unfreeze,
        ) in self.unfreeze_after.items():
            yield node_name, self[node_name], unfreeze_after, lr_after_unfreeze

    def traverse(self) -> Iterator[tuple[str, BaseNode, list[str], list[str]]]:
        yield from traverse_graph(self.graph, self)


def compute_losses(
    cfg: Config,
    losses: dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]],
    loss_weights: dict[str, float],
    device: torch.device,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Computes the final loss as a weighted sum of all the losses.

    @type losses: dict[str, dict[str, Tensor | tuple[Tensor, dict[str,
        Tensor]]]]
    @param losses: Dictionary of computed losses. Each node can have
        multiple losses attached. The first key identifies the node, the
        second key identifies the specific loss. Values are either
        single tensors or tuples of tensors and sub- losses.
    @rtype: tuple[Tensor, dict[str, Tensor]]
    @return: Tuple of final loss and dictionary of all losses for
        logging. The dictionary is in a format of C{{loss_name:
        loss_value}}.
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

            loss *= loss_weights[loss_name]
            final_loss += loss
            all_losses[f"loss/{node_name}/{loss_name}"] = loss.detach().cpu()
            if cfg.trainer.log_sub_losses and sublosses:
                for subloss_name, subloss_value in sublosses.items():
                    all_losses[
                        f"loss/{node_name}/{loss_name}/{subloss_name}"
                    ] = subloss_value.detach().cpu()
    all_losses["loss"] = final_loss.detach().cpu()
    return final_loss, all_losses


def build_losses(
    nodes: Mapping[str, BaseNode], cfg: Config
) -> tuple[AttachedModulesDict[BaseLoss], dict[str, float]]:
    loss_weights = {}
    losses: AttachedModulesDict[BaseLoss] = defaultdict(dict)
    for loss_cfg in cfg.model.losses:
        loss_name, _ = _init_attached_module(nodes, loss_cfg, LOSSES, losses)
        loss_weights[loss_name] = loss_cfg.weight
    return _to_module_dict(losses), loss_weights


def build_visualizers(
    nodes: Mapping[str, BaseNode], cfg: Config
) -> AttachedModulesDict[BaseVisualizer]:
    visualizers: AttachedModulesDict[BaseVisualizer] = defaultdict(dict)
    for visualizer_cfg in cfg.model.visualizers:
        _init_attached_module(nodes, visualizer_cfg, VISUALIZERS, visualizers)
    return _to_module_dict(visualizers)


def build_metrics(
    nodes: Mapping[str, BaseNode], cfg: Config
) -> tuple[AttachedModulesDict[BaseMetric], tuple[str, str] | None]:
    metrics: AttachedModulesDict[BaseMetric] = defaultdict(dict)
    main_metric = None
    for metric_cfg in cfg.model.metrics:
        metric_name, node_name = _init_attached_module(
            nodes, metric_cfg, METRICS, metrics
        )
        if metric_cfg.is_main_metric:
            if main_metric is not None:
                raise ValueError(
                    "Multiple main metrics defined. Only one is allowed."
                )
            if metric_name == "ConfusionMatrix":
                metric_name = "mcc"
            main_metric = (node_name, metric_name)
    return _to_module_dict(metrics), main_metric


def build_training_strategy(
    cfg: Config, pl_module: pl.LightningModule
) -> BaseTrainingStrategy | None:
    training_strategy = cfg.trainer.training_strategy
    if training_strategy is not None:
        logger.info(f"Using training strategy '{training_strategy.name}'")
        if (
            cfg.trainer.optimizer is not None
            or cfg.trainer.scheduler is not None
        ):
            logger.warning(
                "Training strategy is defined. It will override "
                "any specified optimizer or scheduler from the config."
            )
        return from_registry(
            STRATEGIES,
            training_strategy.name,
            **training_strategy.params,
            pl_module=pl_module,
        )
    return None


def build_optimizers(
    cfg: Config,
    parameters: Iterable[nn.Parameter],
    main_metric: tuple[str, str] | None,
    nodes: Nodes,
) -> tuple[list[Optimizer], list[LRScheduler | dict[str, Any]]]:
    """Configures model optimizers and schedulers."""
    cfg_optimizer = cfg.trainer.optimizer
    cfg_scheduler = cfg.trainer.scheduler

    optimizer = from_registry(
        OPTIMIZERS,
        cfg_optimizer.name,
        **cfg_optimizer.params,
        params=[p for p in parameters if p.requires_grad],
    )

    def _get_scheduler(cfg: ConfigItem, optimizer: Optimizer) -> LRScheduler:
        return from_registry(
            SCHEDULERS, cfg.name, **cfg.params, optimizer=optimizer
        )

    if cfg_scheduler.name == "SequentialLR":
        if "schedulers" not in cfg_scheduler.params:
            raise ValueError(
                "'SequentialLR' scheduler requires 'schedulers' "
                "parameter containing the configurations of the "
                "individual schedulers."
            )
        schedulers = cfg_scheduler.params["schedulers"]
        if not check_type(schedulers, list[dict]):
            raise TypeError(
                "'schedulers' parameter of 'SequentialLR' scheduler "
                f"must be a list of dictionaries. Got `{schedulers}`."
            )
        schedulers_list = [
            _get_scheduler(ConfigItem(**scheduler_cfg), optimizer)
            for scheduler_cfg in schedulers
        ]

        if "milestones" not in cfg_scheduler.params:
            raise ValueError(
                "'SequentialLR' scheduler requires 'milestones' parameter."
            )

        milestones = cfg_scheduler.params["milestones"]
        if not check_type(milestones, list[int]):
            raise TypeError(
                "'milestones' parameter of 'SequentialLR' scheduler must be a list of integers. "
                f"Got `{milestones}`."
            )

        scheduler = SequentialLR(
            optimizer, schedulers=schedulers_list, milestones=milestones
        )

    elif cfg_scheduler.name == "ReduceLROnPlateau":
        scheduler = _get_scheduler(cfg_scheduler, optimizer)
        if cfg_scheduler.params.get("mode") == "max":
            if main_metric is None:
                raise ValueError(
                    "ReduceLROnPlateau with 'max' mode requires a main_metric."
                )
            node_name, metric_name = main_metric
            formatted = nodes.formatted_name(node_name)
            monitor = f"val/metric/{formatted}/{metric_name}"
        else:
            monitor = "val/loss"

        scheduler = {
            "scheduler": scheduler,
            "monitor": monitor,
            "frequency": cfg.trainer.validation_interval,
        }

    else:
        scheduler = _get_scheduler(cfg_scheduler, optimizer)

    return [optimizer], [scheduler]


def build_callbacks(
    cfg: Config,
    main_metric: tuple[str, str] | None,
    save_dir: Path,
    nodes: Nodes,
) -> list[pl.Callback]:
    """Configures Pytorch Lightning callbacks."""
    model_name = cfg.model.name

    callbacks: list[pl.Callback] = [
        TrainingManager(),
        LuxonisModelSummary(max_depth=2, rich=cfg.rich_logging),
        ModelCheckpoint(
            dirpath=save_dir / "min_val_loss",
            filename=f"{model_name}_loss={{val/loss:.4f}}_{{epoch:02d}}",
            monitor="val/loss",
            auto_insert_metric_name=False,
            save_top_k=cfg.trainer.save_top_k,
            mode="min",
        ),
    ]

    if main_metric is not None:
        node_name, metric_name = main_metric
        formatted_node = nodes.formatted_name(node_name)
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
                save_top_k=cfg.trainer.save_top_k,
                mode="max",
            )
        )

    for callback in cfg.trainer.callbacks:
        if callback.active:
            callbacks.append(
                from_registry(CALLBACKS, callback.name, **callback.params)
            )
        else:
            logger.info(f"Callback '{callback.name}' is inactive.")

    if cfg.trainer.accumulate_grad_batches is not None:
        if not any(
            isinstance(cb, GradientAccumulationScheduler) for cb in callbacks
        ):
            gas = GradientAccumulationScheduler(
                scheduling={0: cfg.trainer.accumulate_grad_batches}
            )
            callbacks.append(gas)
        else:
            logger.warning(
                "'GradientAccumulationScheduler' is already present "
                "in the callbacks list. The `accumulate_grad_batches` "
                "parameter in the config will be ignored."
            )

    return callbacks


def build_nodes(
    cfg: Config,
    dataset_metadata: DatasetMetadata,
    input_shapes: dict[str, Size],
) -> Nodes:
    frozen_nodes: dict[str, tuple[int, float | None]] = {}
    node_task_names: dict[str, str] = {}
    node_kwargs: dict[str, tuple[type[BaseNode], Kwargs]] = {}
    node_inputs: dict[str, list[str]] = {}
    graph: dict[str, list[str]] = {}
    loader_input_shapes: dict[str, dict[str, Size]] = {}

    for node_cfg in cfg.model.nodes:
        Node = NODES.get(node_cfg.name)
        node_name = node_cfg.alias or node_cfg.name
        if node_cfg.freezing.active:
            epochs = cfg.trainer.epochs
            if node_cfg.freezing.unfreeze_after is None:
                unfreeze_after = epochs
            elif isinstance(node_cfg.freezing.unfreeze_after, int):
                unfreeze_after = node_cfg.freezing.unfreeze_after
            else:
                unfreeze_after = int(node_cfg.freezing.unfreeze_after * epochs)
            frozen_nodes[node_name] = (
                unfreeze_after,
                node_cfg.freezing.lr_after_unfreeze,
            )
        if issubclass(Node, BaseHead):
            task_names = dataset_metadata.task_names
            if node_cfg.task_name is None:
                if len(task_names) == 1:
                    node_cfg.task_name = next(iter(task_names))
                elif len(task_names) > 1:
                    raise ValueError(
                        f"Dataset contains multiple tasks: {task_names}, "
                        f"but node '{node_name}' does not have the "
                        "`task_name` parameter specified. "
                        "Please specify the `task_name` parameter "
                        "for each head node. "
                    )
        node_task_names[node_name] = node_cfg.task_name or ""

        metadata_override = node_cfg.metadata_task_override
        if Node.task is not None:
            metadata = {
                label
                for label in Node.task.required_labels
                if isinstance(label, Metadata)
            }
            if metadata_override is not None:
                if isinstance(metadata_override, str):
                    if len(metadata) != 1:
                        raise ValueError(
                            f"Task '{Node.task}' of node '{Node.__name__}' requires multiple metadata labels: {metadata}, "
                            "so the `metadata_task_override` must be a dictionary."
                        )
                    metadata_override = {
                        next(iter(metadata)).name: metadata_override
                    }

                for m in metadata:
                    m.name = metadata_override.get(m.name, m.name)

            metadata_types = dataset_metadata.metadata_types

            for m in metadata:
                m_name = f"{node_cfg.task_name}/{m}"
                if m_name not in metadata_types:
                    continue
                typ = metadata_types[m_name]
                if not m.check_type(typ):
                    raise ValueError(
                        f"Metadata type mismatch for label '{m}' in node '{node_name}'. "
                        f"Expected type '{m.typ}', got '{typ.__name__}'."
                    )

        node_kwargs[node_name] = (
            Node,
            {
                **node_cfg.params,
                # TODO: `task_name` only makes sense for heads.
                "task_name": node_cfg.task_name or "",
                "remove_on_export": node_cfg.remove_on_export,
                "dataset_metadata": dataset_metadata,
                "original_in_shape": input_shapes[cfg.loader.image_source],
            },
        )

        if node_cfg.input_sources:
            node_inputs[node_name] = node_cfg.input_sources

        if not node_cfg.inputs and not node_cfg.input_sources:
            # If no inputs (= preceding nodes) nor any input_sources (= loader outputs) are specified,
            # assume the node is the starting node and takes all inputs from the loader.

            loader_input_shapes[node_name] = {
                k: Size(v) for k, v in input_shapes.items()
            }
            node_inputs[node_name] = list(input_shapes.keys())
        else:
            # For each input_source, check if the loader provides the required output.
            # If yes, add the shape to the input_shapes dict. If not, raise an error.
            loader_input_shapes[node_name] = {}
            for input_source in node_cfg.input_sources:
                if input_source not in input_shapes:
                    raise ValueError(
                        f"Node {node_name} requires input source {input_source}, "
                        "which is not provided by the loader."
                    )

                loader_input_shapes[node_name][input_source] = Size(
                    input_shapes[input_source]
                )

            # Inputs (= preceding nodes) are handled in the _initiate_nodes method.

        graph[node_name] = node_cfg.inputs

    return Nodes(
        node_kwargs,
        loader_input_shapes,
        node_inputs,
        graph,
        node_task_names,
        frozen_nodes,
    )


def postprocess_metrics(name: str, values: Any) -> dict[str, Tensor]:
    """Convert metric computation result into a dictionary of values."""
    match values:
        case (Tensor(data=value), dict(submetrics)):
            return {name: value} | submetrics
        case Tensor() as value:
            return {name: value}
        case dict(submetrics):
            return submetrics
        case unknown:  # pragma: no cover
            raise ValueError(
                f"Metric '{name}' returned unexpected value of "
                f"type `{type(unknown)}`."
            )


def _init_attached_module(
    nodes: Mapping[str, BaseNode],
    cfg: AttachedModuleConfig,
    registry: Registry,
    storage: AttachedModulesDict,
) -> tuple[str, str]:
    Module = registry.get(cfg.name)
    module_name = cfg.alias or cfg.name
    node_name = cfg.attached_to
    module = Module(**cfg.params, node=nodes[node_name])
    if module_name == "ConfusionMatrix":
        module_name = "mcc"
    storage[node_name][module_name] = module
    return module_name, node_name


def _to_module_dict(modules: AttachedModulesDict[A]) -> AttachedModulesDict[A]:
    return nn.ModuleDict(
        {
            node_name: nn.ModuleDict(node_modules)
            for node_name, node_modules in modules.items()
        }
    )  # type: ignore


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
    """Log images with balanced class distribution."""
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

    for node_name, node_visualizations in visualizations.items():
        node_logged_images = n_logged_images
        formatted_node_name = nodes.formatted_name(node_name)
        for viz_name, viz_batch in node_visualizations.items():
            for idx in logged_indices:
                if node_logged_images >= max_log_images:
                    break
                tracker.log_image(
                    f"{mode}/visualizations/{formatted_node_name}/{viz_name}/{node_logged_images}",
                    viz_batch[idx].detach().cpu().numpy().transpose(1, 2, 0),
                    step=current_epoch,
                )
                node_logged_images += 1

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
    """Log first N images sequentially."""
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
    """Build a buffer of leftover visualizations to fill up to
    `max_log_images` frames.

    @type seq_buffer: list[dict[str, dict[str, Tensor]]]
    @param seq_buffer: Previously buffered visualizations; each item maps node names to
                        dicts of viz names to Tensors of shape [N, …].
    @type visualizations: dict[str, dict[str, Tensor]]
    @param visualizations: Current batch’s visualizations with the same nested structure.
    @type logged_idxs: list[int]
    @param logged_idxs: List of batch indices already logged by the smart (class-balanced) logger.
    @type max_log_images: int
    @param max_log_images: Total number of images we aim to log per epoch.
    @return: A dict `{ node_name: { viz_name: Tensor[...] } }` containing up to the remaining
             number of images needed to reach `max_log_images`, excluding any indices in
             `logged_idxs`. Returns `None` if the buffer is already full or no leftovers exist.
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

    return leftovers if leftovers else None


def get_model_execution_order(
    model: "lxt.LuxonisLightningModule",
) -> list[str]:
    """Get the execution order of the model's nodes."""
    order = []
    handles = []

    for name, module in model.named_modules():
        if name and list(module.parameters()):
            handle = module.register_forward_hook(
                lambda mod, inp, out, n=name: order.append(n)
            )
            handles.append(handle)

    with torch.no_grad():
        dummy_inputs = {
            input_name: torch.zeros(2, *shape, device=model.device)
            for shapes in model.nodes.input_shapes.values()
            for input_name, shape in shapes.items()
        }
        model(dummy_inputs)

    for handle in handles:
        handle.remove()

    return order
