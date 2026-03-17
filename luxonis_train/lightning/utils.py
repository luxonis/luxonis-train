from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar, overload

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
)
from loguru import logger
from luxonis_ml.typing import Kwargs
from luxonis_ml.utils import traverse_graph
from luxonis_ml.utils.registry import Registry
from rich import print
from torch import Size, Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, SequentialLR

import luxonis_train as lxt
from luxonis_train.attached_modules import BaseLoss, BaseMetric, BaseVisualizer
from luxonis_train.attached_modules.base_attached_module import (
    BaseAttachedModule,
)
from luxonis_train.callbacks import LuxonisModelSummary, TrainingManager
from luxonis_train.config import AttachedModuleConfig, Config
from luxonis_train.config.config import (
    FinetuningConfig,
    FinetuningOptimizerConfig,
    FinetuningSchedulerConfig,
    NodeConfig,
    OptimizerConfig,
    ParameterPattern,
    SchedulerConfig,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.heads.base_head import BaseHead
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


class MainMetric(NamedTuple):
    node_name: str
    metric_name: str


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


class NodeWrapper(nn.Module):
    def __init__(
        self,
        name: str,
        module: BaseNode,
        losses: dict[str, BaseLoss],
        metrics: dict[str, BaseMetric],
        visualizers: dict[str, BaseVisualizer],
        unfreeze_after: int | None,
        lr_after_unfreeze: float | None,
        cfg: NodeConfig,
        inputs: list[str] | None = None,
    ):
        super().__init__()
        self.name = name
        self.module = module
        self.losses = _to_module_dict(losses)
        self.metrics = _to_module_dict(metrics)
        self.visualizers = _to_module_dict(visualizers)
        self.unfreeze_after = unfreeze_after
        self.lr_after_unfreeze = lr_after_unfreeze
        self.cfg = cfg
        self.inputs = inputs or []

    @property
    def task_name(self) -> str:
        return self.module.task_name

    @property
    def formatted_name(self) -> str:
        task_name = self.task_name
        return f"{task_name}-{self.name}" if task_name else self.name


class Nodes(dict[str, NodeWrapper] if TYPE_CHECKING else nn.ModuleDict):
    def __init__(
        self,
        cfg: Config,
        dataset_metadata: DatasetMetadata,
        input_shapes: dict[str, Size],
    ):
        self.cfg = cfg
        self.graph: dict[str, list[str]] = {}
        self.nodes: dict[str, NodeWrapper] = {}
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
                cfg=node_cfg,
                inputs=node_input_names,
            )
            node_outputs = node.module.run(node_dummy_inputs)

            dummy_inputs[node_name] = node_outputs
            self.nodes[node_name] = node

        super().__init__(self.nodes)

    @cached_property
    def main_metric_reference(self) -> BaseMetric:
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
                        f"Metadata type mismatch for label '{m}' in node '{node_cfg.identifier}'. "
                        f"Expected type '{m.typ}', got '{typ.__name__}'."
                    )

    def _extract_optimizer_params(
        self,
    ) -> Iterable[tuple[OptimizerConfig, SchedulerConfig]]:
        cfg_base_optimizer = self.cfg.trainer.optimizer
        cfg_base_scheduler = self.cfg.trainer.scheduler
        groups: dict[str, tuple[list[Kwargs], SchedulerConfig]] = {}
        used_params = set()
        for node in self.values():
            for finetuning in [
                *node.cfg.finetuning,
                FinetuningConfig(
                    parameters=[ParameterPattern(name=".*")],
                    optimizer=cfg_base_optimizer.to_finetuning(),
                    scheduler=cfg_base_scheduler.to_finetuning(),
                ),
            ]:
                cfg_optimizer = merge_config_items(
                    cfg_base_optimizer, finetuning.optimizer
                )
                cfg_scheduler = merge_config_items(
                    cfg_base_scheduler, finetuning.scheduler
                )
                params = []
                for module_name, module in node.module.named_modules():
                    if list(module.parameters()) and not list(
                        module.children()
                    ):
                        for param_name, p in module.named_parameters():
                            name = f"{module.__class__.__name__}.{module_name}.{param_name}"
                            print(name)
                            if (
                                finetuning.parameter_regex.search(name)
                                and p.requires_grad
                                and id(p) not in used_params
                            ):
                                params.append(p)
                                used_params.add(id(p))

                if params:
                    if cfg_optimizer.name not in groups:
                        groups[cfg_optimizer.name] = (
                            [],
                            cfg_scheduler,
                        )
                    groups[cfg_optimizer.name][0].append(
                        {"params": params} | cfg_optimizer.params
                    )

        for optimizer_name, (optimizer_params, scheduler) in groups.items():
            yield (
                OptimizerConfig(
                    name=optimizer_name,
                    params={"params": optimizer_params},  # type: ignore
                ),
                scheduler,
            )

    def _get_freezing(
        self, node_cfg: NodeConfig, total_epochs: int
    ) -> tuple[int | None, float | None]:
        unfreeze_after = None
        lr_after_unfreeze = None
        if node_cfg.freezing.active:
            if node_cfg.freezing.unfreeze_after is None:
                unfreeze_after = total_epochs
            elif isinstance(node_cfg.freezing.unfreeze_after, int):
                unfreeze_after = node_cfg.freezing.unfreeze_after
            else:
                unfreeze_after = int(
                    node_cfg.freezing.unfreeze_after * total_epochs
                )
            if node_cfg.freezing.lr_after_unfreeze is not None:
                lr_after_unfreeze = node_cfg.freezing.lr_after_unfreeze
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
        return self[node_name].formatted_name

    def frozen_nodes(
        self,
    ) -> Iterator[tuple[str, BaseNode, int, float | None]]:
        for node_name, node in self.items():
            if node.unfreeze_after is not None:
                yield (
                    node_name,
                    node.module,
                    node.unfreeze_after,
                    node.lr_after_unfreeze,
                )

    def traverse(
        self,
    ) -> Iterator[tuple[str, NodeWrapper, list[str], list[str]]]:
        yield from traverse_graph(self.graph, self)

    def build_optimizers(
        self,
    ) -> tuple[
        Sequence[Optimizer],
        Sequence[LRSchedulerTypeUnion | LRSchedulerConfig],
    ]:

        optimizers = []
        schedulers = []

        for cfg_optimizer, cfg_scheduler in self._extract_optimizer_params():
            optimizer, scheduler = build_optimizer_scheduler(
                self.main_metric,
                cfg_optimizer,
                cfg_scheduler,
                self.cfg.trainer.validation_interval,
            )
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        return optimizers, schedulers

    def build_callbacks(self, save_dir: Path) -> list[pl.Callback]:
        """Configures Pytorch Lightning callbacks."""
        model_name = self.cfg.model.name

        callbacks: list[pl.Callback] = [
            TrainingManager(),
            LuxonisModelSummary(max_depth=2, rich=self.cfg.rich_logging),
            ModelCheckpoint(
                dirpath=save_dir / "min_val_loss",
                filename=f"{model_name}_loss={{val/loss:.4f}}_{{epoch:02d}}",
                monitor="val/loss",
                auto_insert_metric_name=False,
                save_top_k=self.cfg.trainer.save_top_k,
                mode="min",
            ),
        ]

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
                    save_top_k=self.cfg.trainer.save_top_k,
                    mode="max",
                )
            )

        optimizers, _ = self.build_optimizers()

        for callback in self.cfg.trainer.callbacks:
            if callback.active:
                if (
                    callback.name == "GradientAccumulationScheduler"
                    and len(optimizers) > 1
                ):
                    logger.warning(
                        "Gradient accumulation scheduling is not supported for multiple optimizers. "
                        "The callback will be ignored."
                    )
                    continue
                callbacks.append(
                    from_registry(CALLBACKS, callback.name, **callback.params)
                )
            else:
                logger.info(f"Callback '{callback.name}' is inactive.")

        if self.cfg.trainer.accumulate_grad_batches is not None:
            if len(optimizers) > 1:
                logger.warning(
                    "Gradient accumulation scheduling is not supported for multiple optimizers. "
                    "The `accumulate_grad_batches` parameter in the config will be ignored."
                )
                return callbacks
            if not any(
                isinstance(cb, GradientAccumulationScheduler)
                for cb in callbacks
            ):
                gas = GradientAccumulationScheduler(
                    scheduling={0: self.cfg.trainer.accumulate_grad_batches}
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


def postprocess_metrics(
    name: str, values: Any, log_sub_metrics: bool = True
) -> dict[str, Tensor]:
    """Convert metric computation result into a dictionary of values."""
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


T = TypeVar("T", bound=BaseAttachedModule)


def _init_attached_module(
    node: BaseNode,
    cfg: AttachedModuleConfig,
    registry: Registry[type[T]],
    **kwargs,
) -> tuple[str, T]:
    Module = registry.get(cfg.name)
    module_name = cfg.identifier
    module = Module(**cfg.params, node=node, **kwargs)
    if module_name == "ConfusionMatrix":
        module_name = "mcc"
    return module_name, module


A = TypeVar("A", BaseLoss, BaseMetric, BaseVisualizer)


def _to_module_dict(modules: dict[str, A]) -> dict[str, A]:
    return nn.ModuleDict(modules)  # type: ignore


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
    @param visualizations: Current batch's visualizations with the same nested structure.
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

    return leftovers or None


def get_model_execution_order(
    model: "lxt.LuxonisLightningModule",
) -> list[str]:
    """Get the execution order of the model's nodes."""
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
    """Return whether a tensor (or every tensor in a sequence) resides
    on a given device."""
    if isinstance(x, Tensor):
        return x.device == device
    if isinstance(x, (list | tuple)):
        return all(isinstance(i, Tensor) and i.device == device for i in x)
    raise TypeError(f"Expected Tensor or list[Tensor], got {type(x)!r}")


@overload
def merge_config_items(
    base: OptimizerConfig,
    override: FinetuningOptimizerConfig | FinetuningSchedulerConfig | None,
) -> OptimizerConfig: ...


@overload
def merge_config_items(
    base: SchedulerConfig,
    override: FinetuningOptimizerConfig | FinetuningSchedulerConfig | None,
) -> SchedulerConfig: ...


def merge_config_items(
    base: OptimizerConfig | SchedulerConfig,
    override: FinetuningOptimizerConfig | FinetuningSchedulerConfig | None,
) -> OptimizerConfig | SchedulerConfig:
    if override is None:
        return base.to_finetuning()

    if override.name is None or override.name == base.name:
        name = base.name
        params = base.params | override.params
    else:
        name = override.name
        params = override.params

    return type(base)(name=name, params=params)


def build_optimizer_scheduler(
    main_metric: MainMetric | None,
    cfg_optimizer: OptimizerConfig,
    cfg_scheduler: SchedulerConfig,
    validation_interval: int,
) -> tuple[Optimizer, LRScheduler | LRSchedulerConfigType]:
    """Configures model optimizers and schedulers."""

    optimizer = from_registry(
        OPTIMIZERS, cfg_optimizer.name, **cfg_optimizer.params
    )
    scheduler: LRScheduler | LRSchedulerConfigType

    def _get_scheduler(
        cfg: SchedulerConfig, optimizer: Optimizer
    ) -> LRScheduler:
        return from_registry(
            SCHEDULERS, cfg.name, **cfg.params, optimizer=optimizer
        )

    if cfg_scheduler.name == "SequentialLR":
        scheduler_params = cfg_scheduler.get_sequential_lr_params()

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                _get_scheduler(scheduler_cfg, optimizer)
                for scheduler_cfg in scheduler_params.schedulers
            ],
            milestones=scheduler_params.milestones,
            last_epoch=scheduler_params.last_epoch,
        )

    elif cfg_scheduler.name == "ReduceLROnPlateau":
        reduce_scheduler = _get_scheduler(cfg_scheduler, optimizer)
        if cfg_scheduler.params.get("mode") == "max":
            if main_metric is None:
                raise ValueError(
                    "ReduceLROnPlateau with 'max' mode "
                    "requires a metric to monitor."
                )
            monitor = (
                f"val/metric/{main_metric.node_name}/{main_metric.metric_name}"
            )
        else:
            monitor = "val/loss"

        scheduler = {
            "scheduler": reduce_scheduler,
            "monitor": monitor,
            "frequency": validation_interval,
        }

    else:
        scheduler = _get_scheduler(cfg_scheduler, optimizer)

    return optimizer, scheduler
