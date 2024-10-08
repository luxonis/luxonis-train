from collections import defaultdict
from collections.abc import Mapping
from logging import getLogger
from pathlib import Path
from typing import Literal, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.utilities import rank_zero_only  # type: ignore
from luxonis_ml.data import LuxonisDataset
from torch import Size, Tensor, nn

import luxonis_train
from luxonis_train.attached_modules import (
    BaseAttachedModule,
    BaseLoss,
    BaseMetric,
    BaseVisualizer,
)
from luxonis_train.attached_modules.metrics.torchmetrics import (
    TorchMetricWrapper,
)
from luxonis_train.attached_modules.visualizers import (
    combine_visualizations,
    get_denormalized_images,
)
from luxonis_train.callbacks import BaseLuxonisProgressBar, ModuleFreezer
from luxonis_train.config import AttachedModuleConfig, Config
from luxonis_train.nodes import BaseNode
from luxonis_train.utils import (
    DatasetMetadata,
    Kwargs,
    Labels,
    LuxonisTrackerPL,
    Packet,
    to_shape_packet,
    traverse_graph,
)
from luxonis_train.utils.graph import Graph
from luxonis_train.utils.registry import (
    CALLBACKS,
    OPTIMIZERS,
    SCHEDULERS,
    Registry,
)

from .luxonis_output import LuxonisOutput

logger = getLogger(__name__)


class LuxonisLightningModule(pl.LightningModule):
    """Class representing the entire model.

    This class keeps track of the model graph, nodes, and attached modules.
    The model topology is defined as an acyclic graph of nodes.
    The graph is saved as a dictionary of predecessors.

    @type save_dir: str
    @ivar save_dir: Directory to save checkpoints and logs.

    @type nodes: L{nn.ModuleDict}[str, L{LuxonisModule}]
    @ivar nodes: Nodes of the model. Keys are node names, unique for each node.

    @type graph: dict[str, list[str]]
    @ivar graph: Graph of the model in a format of a dictionary of predecessors.
        Keys are node names, values are inputs to the node (list of node names).
        Nodes with no inputs are considered inputs of the whole model.

    @type loss_weights: dict[str, float]
    @ivar loss_weights: Dictionary of loss weights. Keys are loss names, values are weights.

    @type input_shapes: dict[str, list[L{Size}]]
    @ivar input_shapes: Dictionary of input shapes. Keys are node names, values are lists of shapes
        (understood as shapes of the "feature" field in L{Packet}[L{Tensor}]).

    @type outputs: list[str]
    @ivar outputs: List of output node names.

    @type losses: L{nn.ModuleDict}[str, L{nn.ModuleDict}[str, L{LuxonisLoss}]]
    @ivar losses: Nested dictionary of losses used in the model. Each node can have multiple
        losses attached. The first key identifies the node, the second key identifies the
        specific loss.

    @type visualizers: dict[str, dict[str, L{LuxonisVisualizer}]]
    @ivar visualizers: Dictionary of visualizers to be used with the model.

    @type metrics: dict[str, dict[str, L{LuxonisMetric}]]
    @ivar metrics: Dictionary of metrics to be used with the model.

    @type dataset_metadata: L{DatasetMetadata}
    @ivar dataset_metadata: Metadata of the dataset.

    @type main_metric: str | None
    @ivar main_metric: Name of the main metric to be used for model checkpointing.
        If not set, the model with the best metric score won't be saved.
    """

    _trainer: pl.Trainer
    logger: LuxonisTrackerPL

    def __init__(
        self,
        cfg: Config,
        save_dir: str,
        input_shapes: dict[str, Size],
        dataset_metadata: DatasetMetadata | None = None,
        *,
        _core: "luxonis_train.core.LuxonisModel | None" = None,
        **kwargs,
    ):
        """Constructs an instance of `LuxonisModel` from `Config`.

        @type cfg: L{Config}
        @param cfg: Config object.
        @type save_dir: str
        @param save_dir: Directory to save checkpoints.
        @type input_shapes: dict[str, Size]
        @param input_shapes: Dictionary of input shapes. Keys are input
            names, values are shapes.
        @type dataset_metadata: L{DatasetMetadata} | None
        @param dataset_metadata: Dataset metadata.
        @type kwargs: Any
        @param kwargs: Additional arguments to pass to the
            L{LightningModule} constructor.
        """
        super().__init__(**kwargs)

        self._export: bool = False
        self._core = _core

        self.cfg = cfg
        self.original_in_shapes = input_shapes
        self.image_source = cfg.loader.image_source
        self.dataset_metadata = dataset_metadata or DatasetMetadata()
        self.frozen_nodes: list[tuple[nn.Module, int]] = []
        self.graph: Graph = {}
        self.loader_input_shapes: dict[str, dict[str, Size]] = {}
        self.node_input_sources: dict[str, list[str]] = defaultdict(list)
        self.loss_weights: dict[str, float] = {}
        self.main_metric: str | None = None
        self.save_dir = save_dir
        self.test_step_outputs: list[Mapping[str, Tensor | float | int]] = []
        self.training_step_outputs: list[
            Mapping[str, Tensor | float | int]
        ] = []
        self.validation_step_outputs: list[
            Mapping[str, Tensor | float | int]
        ] = []
        self.losses: dict[str, dict[str, BaseLoss]] = defaultdict(dict)
        self.metrics: dict[str, dict[str, BaseMetric]] = defaultdict(dict)
        self.visualizers: dict[str, dict[str, BaseVisualizer]] = defaultdict(
            dict
        )

        self._logged_images = 0

        frozen_nodes: list[tuple[str, int]] = []
        nodes: dict[str, tuple[type[BaseNode], Kwargs]] = {}

        for node_cfg in self.cfg.model.nodes:
            node_name = node_cfg.name
            Node: type[BaseNode] = BaseNode.REGISTRY.get(node_name)
            node_name = node_cfg.alias or node_name
            if node_cfg.freezing.active:
                epochs = self.cfg.trainer.epochs
                if node_cfg.freezing.unfreeze_after is None:
                    unfreeze_after = epochs
                elif isinstance(node_cfg.freezing.unfreeze_after, int):
                    unfreeze_after = node_cfg.freezing.unfreeze_after
                else:
                    unfreeze_after = int(
                        node_cfg.freezing.unfreeze_after * epochs
                    )
                frozen_nodes.append((node_name, unfreeze_after))

            if node_cfg.task is not None:
                if Node.tasks is None:
                    raise ValueError(
                        f"Cannot define tasks for node {node_name}."
                        "This node doesn't specify any tasks."
                    )
                if isinstance(node_cfg.task, str):
                    assert Node.tasks
                    if len(Node.tasks) > 1:
                        raise ValueError(
                            f"Node {node_name} specifies multiple tasks, "
                            "but only one task is specified in the config. "
                            "Specify the tasks as a dictionary instead."
                        )

                    node_cfg.task = {next(iter(Node.tasks)): node_cfg.task}
                else:
                    node_cfg.task = {
                        **Node._process_tasks(Node.tasks),
                        **node_cfg.task,
                    }
            nodes[node_name] = (
                Node,
                {
                    **node_cfg.params,
                    "_tasks": node_cfg.task,
                    "remove_on_export": node_cfg.remove_on_export,
                },
            )

            # Handle inputs for this node
            if node_cfg.input_sources:
                self.node_input_sources[node_name] = node_cfg.input_sources

            if not node_cfg.inputs and not node_cfg.input_sources:
                # If no inputs (= preceding nodes) nor any input_sources (= loader outputs) are specified,
                # assume the node is the starting node and takes all inputs from the loader.

                self.loader_input_shapes[node_name] = {
                    k: Size(v) for k, v in input_shapes.items()
                }
                self.node_input_sources[node_name] = list(input_shapes.keys())
            else:
                # For each input_source, check if the loader provides the required output.
                # If yes, add the shape to the input_shapes dict. If not, raise an error.
                self.loader_input_shapes[node_name] = {}
                for input_source in node_cfg.input_sources:
                    if input_source not in input_shapes:
                        raise ValueError(
                            f"Node {node_name} requires input source {input_source}, "
                            "which is not provided by the loader."
                        )

                    self.loader_input_shapes[node_name][input_source] = Size(
                        input_shapes[input_source]
                    )

                # Inputs (= preceding nodes) are handled in the _initiate_nodes method.

            self.graph[node_name] = node_cfg.inputs

        self.nodes = self._initiate_nodes(nodes)

        for loss_cfg in self.cfg.model.losses:
            loss_name, _ = self._init_attached_module(
                loss_cfg, BaseLoss.REGISTRY, self.losses
            )
            self.loss_weights[loss_name] = loss_cfg.weight

        for metric_cfg in self.cfg.model.metrics:
            metric_name, node_name = self._init_attached_module(
                metric_cfg, BaseMetric.REGISTRY, self.metrics
            )
            if metric_cfg.is_main_metric:
                if self.main_metric is not None:
                    raise ValueError(
                        "Multiple main metrics defined. Only one is allowed."
                    )
                self.main_metric = f"{node_name}/{metric_name}"

        for visualizer_cfg in self.cfg.model.visualizers:
            self._init_attached_module(
                visualizer_cfg, BaseVisualizer.REGISTRY, self.visualizers
            )

        self.outputs = self.cfg.model.outputs
        self.frozen_nodes = [(self.nodes[name], e) for name, e in frozen_nodes]
        self.losses = self._to_module_dict(self.losses)  # type: ignore
        self.metrics = self._to_module_dict(self.metrics)  # type: ignore
        self.visualizers = self._to_module_dict(self.visualizers)  # type: ignore

        self.load_checkpoint(self.cfg.model.weights)

    @property
    def core(self) -> "luxonis_train.core.LuxonisModel":
        """Returns the core model."""
        if self._core is None:  # pragma: no cover
            raise ValueError("Core reference is not set.")
        return self._core

    def _initiate_nodes(
        self,
        nodes: dict[str, tuple[type[BaseNode], Kwargs]],
    ) -> nn.ModuleDict:
        """Initializes all the nodes in the model.

        Traverses the graph and initiates each node using outputs of the
        preceding nodes.

        @type nodes: dict[str, tuple[type[LuxonisNode], Kwargs]]
        @param nodes: Dictionary of nodes to be initiated. Keys are node
            names, values are tuples of node class and node kwargs.
        @rtype: L{nn.ModuleDict}[str, L{LuxonisNode}]
        @return: Dictionary of initiated nodes.
        """
        initiated_nodes: dict[str, BaseNode] = {}

        dummy_inputs: dict[str, Packet[Tensor]] = {
            source_name: {"features": [torch.zeros(2, *shape)]}
            for shapes in self.loader_input_shapes.values()
            for source_name, shape in shapes.items()
        }

        for (
            node_name,
            (
                Node,
                node_kwargs,
            ),
            node_input_names,
            _,
        ) in traverse_graph(self.graph, nodes):
            node_dummy_inputs: list[Packet[Tensor]] = []
            """List of dummy input packets for the node.

            The first one is always from the loader.
            """
            node_input_shapes: list[Packet[Size]] = []
            """Corresponding list of input shapes."""

            node_input_names += self.node_input_sources[node_name]
            for node_input_name in node_input_names:
                dummy_input = dummy_inputs[node_input_name]

                node_dummy_inputs.append(dummy_input)

                shape_packet = to_shape_packet(dummy_input)
                node_input_shapes.append(shape_packet)

            node = Node(
                input_shapes=node_input_shapes,
                original_in_shape=self.original_in_shapes[self.image_source],
                dataset_metadata=self.dataset_metadata,
                **node_kwargs,
            )
            node_outputs = node.run(node_dummy_inputs)

            dummy_inputs[node_name] = node_outputs
            initiated_nodes[node_name] = node

        return nn.ModuleDict(initiated_nodes)

    def forward(
        self,
        inputs: dict[str, Tensor],
        labels: Labels | None = None,
        images: Tensor | None = None,
        *,
        compute_loss: bool = True,
        compute_metrics: bool = False,
        compute_visualizations: bool = False,
    ) -> LuxonisOutput:
        """Forward pass of the model.

        Traverses the graph and step-by-step computes the outputs of
        each node. Each next node is computed only when all of its
        predecessors are computed. Once the outputs are not needed
        anymore, they are removed from the memory.

        @type inputs: L{Tensor}
        @param inputs: Input tensor.
        @type task_labels: L{TaskLabels} | None
        @param task_labels: Labels dictionary. Defaults to C{None}.
        @type images: L{Tensor} | None
        @param images: Canvas tensor for visualizers. Defaults to
            C{None}.
        @type compute_loss: bool
        @param compute_loss: Whether to compute losses. Defaults to
            C{True}.
        @type compute_metrics: bool
        @param compute_metrics: Whether to update metrics. Defaults to
            C{True}.
        @type compute_visualizations: bool
        @param compute_visualizations: Whether to compute
            visualizations. Defaults to C{False}.
        @rtype: L{LuxonisOutput}
        @return: Output of the model.
        """
        losses: dict[
            str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]
        ] = defaultdict(dict)
        visualizations: dict[str, dict[str, Tensor]] = defaultdict(dict)

        computed: dict[str, Packet[Tensor]] = {}
        for node_name, node, input_names, unprocessed in traverse_graph(
            self.graph, cast(dict[str, BaseNode], self.nodes)
        ):
            if node.export and node.remove_on_export:
                continue
            input_names += self.node_input_sources[node_name]

            node_inputs: list[Packet[Tensor]] = []
            for pred in input_names:
                if pred in computed:
                    node_inputs.append(computed[pred])
                else:
                    node_inputs.append({"features": [inputs[pred]]})
            outputs = node.run(node_inputs)
            computed[node_name] = outputs

            if (
                compute_loss
                and node_name in self.losses
                and labels is not None
            ):
                for loss_name, loss in self.losses[node_name].items():
                    losses[node_name][loss_name] = loss.run(outputs, labels)

            if (
                compute_metrics
                and node_name in self.metrics
                and labels is not None
            ):
                for metric in self.metrics[node_name].values():
                    metric.run_update(outputs, labels)

            if (
                compute_visualizations
                and node_name in self.visualizers
                and images is not None
            ):
                for viz_name, visualizer in self.visualizers[
                    node_name
                ].items():
                    viz = combine_visualizations(
                        visualizer.run(
                            images,
                            images,
                            outputs,
                            labels,
                        ),
                    )
                    visualizations[node_name][viz_name] = viz

            for computed_name in list(computed.keys()):
                if computed_name in self.outputs:
                    continue
                for node_name in unprocessed:
                    if computed_name in self.graph[node_name]:
                        break
                else:
                    del computed[computed_name]

        outputs_dict = {
            node_name: outputs
            for node_name, outputs in computed.items()
            if node_name in self.outputs
        }

        return LuxonisOutput(
            outputs=outputs_dict, losses=losses, visualizations=visualizations
        )

    def compute_metrics(self) -> dict[str, dict[str, Tensor]]:
        """Computes metrics and returns their values.

        Goes through all metrics in the `metrics` attribute and computes their values.
        After the computation, the metrics are reset.

        @rtype: dict[str, dict[str, L{Tensor}]]
        @return: Dictionary of computed metrics. Each node can have multiple metrics
            attached. The first key identifies the node, the second key identifies
            the specific metric.
        """
        metric_results: dict[str, dict[str, Tensor]] = defaultdict(dict)
        for node_name, metrics in self.metrics.items():
            for metric_name, metric in metrics.items():
                match metric.compute():
                    case (Tensor(data=metric_value), dict(submetrics)):
                        computed_submetrics = {
                            metric_name: metric_value,
                        } | submetrics
                    case Tensor() as metric_value:
                        computed_submetrics = {metric_name: metric_value}
                    case dict(submetrics):
                        computed_submetrics = submetrics
                    case unknown:  # pragma: no cover
                        raise ValueError(
                            f"Metric {metric_name} returned unexpected value of "
                            f"type {type(unknown)}."
                        )
                metric.reset()
                metric_results[node_name] |= computed_submetrics
        return metric_results

    def export_onnx(self, save_path: str, **kwargs) -> list[str]:
        """Exports the model to ONNX format.

        @type save_path: str
        @param save_path: Path where the exported model will be saved.
        @type kwargs: Any
        @param kwargs: Additional arguments for the L{torch.onnx.export}
            method.
        @rtype: list[str]
        @return: List of output names.
        """
        self.eval()

        inputs = {
            input_name: torch.zeros([1, *shape]).to(self.device)
            for shapes in self.loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }

        inputs_deep_clone = {
            k: torch.zeros(elem.shape).to(self.device)
            for k, elem in inputs.items()
        }

        inputs_for_onnx = {"inputs": inputs_deep_clone}

        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode()

        outputs = self.forward(inputs_deep_clone).outputs
        output_order = sorted(
            [
                (node_name, output_name, i)
                for node_name, outs in outputs.items()
                for output_name, out in outs.items()
                for i in range(len(out))
            ]
        )

        if self.cfg.exporter.output_names is not None:
            len_names = len(self.cfg.exporter.output_names)
            if len_names != len(output_order):
                logger.warning(
                    f"Number of provided output names ({len_names}) does not match "
                    f"number of outputs ({len(output_order)}). Using default names."
                )
                self.cfg.exporter.output_names = None

        output_names = self.cfg.exporter.output_names or [
            f"{node_name}/{output_name}/{i}"
            for node_name, output_name, i in output_order
        ]

        if not self.cfg.exporter.output_names:
            idx = 1
            # Set to output names required by DAI
            for i, output_name in enumerate(output_names):
                if output_name.startswith("EfficientBBoxHead"):
                    output_names[i] = f"output{idx}_yolov6r2"
                    idx += 1

        old_forward = self.forward

        def export_forward(inputs) -> tuple[Tensor, ...]:
            outputs = old_forward(
                inputs,
                None,
                compute_loss=False,
                compute_metrics=False,
                compute_visualizations=False,
            ).outputs
            return tuple(
                outputs[node_name][output_name][i]
                for node_name, output_name, i in output_order
            )

        self.forward = export_forward  # type: ignore

        if "input_names" not in kwargs:
            kwargs["input_names"] = list(inputs.keys())
        if "output_names" not in kwargs:
            kwargs["output_names"] = output_names

        self.to_onnx(save_path, inputs_for_onnx, **kwargs)

        self.forward = old_forward  # type: ignore

        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode(False)

        logger.info(f"Model exported to {save_path}")

        self.train()

        return output_names

    def process_losses(
        self,
        losses_dict: dict[
            str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]
        ],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Processes individual losses from the model run.

        Goes over the computed losses and computes the final loss as a
        weighted sum of all the losses.

        @type losses_dict: dict[str, dict[str, Tensor | tuple[Tensor,
            dict[str, Tensor]]]]
        @param losses_dict: Dictionary of computed losses. Each node can
            have multiple losses attached. The first key identifies the
            node, the second key identifies the specific loss. Values
            are either single tensors or tuples of tensors and sub-
            losses.
        @rtype: tuple[Tensor, dict[str, Tensor]]
        @return: Tuple of final loss and dictionary of processed sub-
            losses. The dictionary is in a format of {loss_name:
            loss_value}.
        """
        final_loss = torch.zeros(1, device=self.device)
        training_step_output: dict[str, Tensor] = {}
        for node_name, losses in losses_dict.items():
            for loss_name, loss_values in losses.items():
                if isinstance(loss_values, tuple):
                    loss, sublosses = loss_values
                else:
                    loss = loss_values
                    sublosses = {}

                loss *= self.loss_weights[loss_name]
                final_loss += loss
                training_step_output[f"loss/{node_name}/{loss_name}"] = (
                    loss.detach().cpu()
                )
                if self.cfg.trainer.log_sub_losses and sublosses:
                    for subloss_name, subloss_value in sublosses.items():
                        training_step_output[
                            f"loss/{node_name}/{loss_name}/{subloss_name}"
                        ] = subloss_value.detach().cpu()
        training_step_output["loss"] = final_loss.detach().cpu()
        return final_loss, training_step_output

    def training_step(
        self, train_batch: tuple[dict[str, Tensor], Labels]
    ) -> Tensor:
        """Performs one step of training with provided batch."""
        outputs = self.forward(*train_batch)
        assert (
            outputs.losses
        ), "Losses are empty, check if you have defined any loss"

        loss, training_step_output = self.process_losses(outputs.losses)
        self.training_step_outputs.append(training_step_output)
        return loss

    def validation_step(
        self, val_batch: tuple[dict[str, Tensor], Labels]
    ) -> dict[str, Tensor]:
        """Performs one step of validation with provided batch."""
        return self._evaluation_step("val", val_batch)

    def test_step(
        self, test_batch: tuple[dict[str, Tensor], Labels]
    ) -> dict[str, Tensor]:
        """Performs one step of testing with provided batch."""
        return self._evaluation_step("test", test_batch)

    def on_train_epoch_end(self) -> None:
        """Performs train epoch end operations."""
        epoch_train_losses = self._average_losses(self.training_step_outputs)
        for module in self.modules():
            if isinstance(module, (BaseNode, BaseLoss)):
                module._epoch = self.current_epoch

        for key, value in epoch_train_losses.items():
            self.log(f"train/{key}", value, sync_dist=True)

        self.training_step_outputs.clear()

    def on_validation_epoch_end(self) -> None:
        """Performs validation epoch end operations."""
        return self._evaluation_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        """Performs test epoch end operations."""
        return self._evaluation_epoch_end("test")

    def get_status(self) -> tuple[int, int]:
        """Returns current epoch and number of all epochs."""
        return self.current_epoch, self.cfg.trainer.epochs

    def get_status_percentage(self) -> float:
        """Returns percentage of current training, takes into account
        early stopping."""
        if self._trainer.early_stopping_callback:
            # model haven't yet stop from early stopping callback
            if self._trainer.early_stopping_callback.stopped_epoch == 0:
                return (self.current_epoch / self.cfg.trainer.epochs) * 100
            else:
                return 100.0
        else:
            return (self.current_epoch / self.cfg.trainer.epochs) * 100

    def _evaluation_step(
        self,
        mode: Literal["test", "val"],
        batch: tuple[dict[str, Tensor], Labels],
    ) -> dict[str, Tensor]:
        inputs, labels = batch
        images = None
        if self._logged_images < self.cfg.trainer.n_log_images:
            images = get_denormalized_images(self.cfg, inputs)
        outputs = self.forward(
            inputs,
            labels,
            images=images,
            compute_metrics=True,
            compute_visualizations=True,
        )

        _, step_output = self.process_losses(outputs.losses)
        self.validation_step_outputs.append(step_output)

        logged_images = self._logged_images
        for node_name, visualizations in outputs.visualizations.items():
            for viz_name, viz_batch in visualizations.items():
                logged_images = self._logged_images
                for viz in viz_batch:
                    if logged_images >= self.cfg.trainer.n_log_images:
                        break
                    self.logger.log_image(
                        f"{mode}/visualizations/{node_name}/{viz_name}/{logged_images}",
                        viz.detach().cpu().numpy().transpose(1, 2, 0),
                        step=self.current_epoch,
                    )
                    logged_images += 1
        self._logged_images = logged_images

        return step_output

    def _evaluation_epoch_end(self, mode: Literal["test", "val"]) -> None:
        epoch_val_losses = self._average_losses(self.validation_step_outputs)

        for key, value in epoch_val_losses.items():
            self.log(f"{mode}/{key}", value, sync_dist=True)

        metric_results: dict[str, dict[str, float]] = defaultdict(dict)
        logger.info(f"Computing metrics on {mode} subset ...")
        computed_metrics = self.compute_metrics()
        logger.info("Metrics computed.")
        for node_name, metrics in computed_metrics.items():
            for metric_name, metric_value in metrics.items():
                metric_results[node_name][metric_name] = (
                    metric_value.cpu().item()
                )
                self.log(
                    f"{mode}/metric/{node_name}/{metric_name}",
                    metric_value,
                    sync_dist=True,
                )

        if self.cfg.trainer.verbose:
            self._print_results(
                stage="Validation" if mode == "val" else "Test",
                loss=epoch_val_losses["loss"],
                metrics=metric_results,
            )

        self.validation_step_outputs.clear()
        self._logged_images = 0

    def configure_callbacks(self) -> list[pl.Callback]:
        """Configures Pytorch Lightning callbacks."""
        self.min_val_loss_checkpoints_path = f"{self.save_dir}/min_val_loss"
        self.best_val_metric_checkpoints_path = (
            f"{self.save_dir}/best_val_metric"
        )
        model_name = self.cfg.model.name

        callbacks: list[pl.Callback] = [
            ModelCheckpoint(
                monitor="val/loss",
                dirpath=self.min_val_loss_checkpoints_path,
                filename=f"{model_name}_loss={{val/loss:.4f}}_{{epoch:02d}}",
                auto_insert_metric_name=False,
                save_top_k=self.cfg.trainer.save_top_k,
                mode="min",
            ),
            RichModelSummary(max_depth=2),
        ]

        if self.main_metric is not None:
            main_metric = self.main_metric.replace("/", "_")
            callbacks.append(
                ModelCheckpoint(
                    monitor=f"val/metric/{self.main_metric}",
                    dirpath=self.best_val_metric_checkpoints_path,
                    filename=f"{model_name}_{main_metric}={{val/metric/{self.main_metric}:.4f}}"
                    f"_loss={{val/loss:.4f}}_{{epoch:02d}}",
                    auto_insert_metric_name=False,
                    save_top_k=self.cfg.trainer.save_top_k,
                    mode="max",
                )
            )

        if self.frozen_nodes:
            callbacks.append(ModuleFreezer(self.frozen_nodes))

        for callback in self.cfg.trainer.callbacks:
            if callback.active:
                callbacks.append(
                    CALLBACKS.get(callback.name)(**callback.params)
                )

        return callbacks

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler],
    ]:
        """Configures model optimizers and schedulers."""
        cfg_optimizer = self.cfg.trainer.optimizer
        cfg_scheduler = self.cfg.trainer.scheduler

        optim_params = cfg_optimizer.params | {
            "params": filter(lambda p: p.requires_grad, self.parameters()),
        }
        optimizer = OPTIMIZERS.get(cfg_optimizer.name)(**optim_params)

        scheduler_params = cfg_scheduler.params | {"optimizer": optimizer}
        scheduler = SCHEDULERS.get(cfg_scheduler.name)(**scheduler_params)

        return [optimizer], [scheduler]

    def load_checkpoint(self, path: str | Path | None) -> None:
        """Loads checkpoint weights from provided path.

        Loads the checkpoints gracefully, ignoring keys that are not
        found in the model state dict or in the checkpoint.

        @type path: str | None
        @param path: Path to the checkpoint. If C{None}, no checkpoint
            will be loaded.
        """
        if path is None:
            return

        path = str(path)
        checkpoint = torch.load(path, map_location=self.device)

        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain state_dict.")
        state_dict = {}
        self_state_dict = self.state_dict()
        for key, value in checkpoint["state_dict"].items():
            if key not in self_state_dict.keys():
                logger.warning(
                    f"Key `{key}` from checkpoint not found in model state dict."
                )
            else:
                state_dict[key] = value

        for key in self_state_dict:
            if key not in state_dict.keys():
                logger.warning(f"Key `{key}` was not found in checkpoint.")
            else:
                try:
                    self_state_dict[key].copy_(state_dict[key])
                except Exception:
                    logger.warning(
                        f"Key `{key}` from checkpoint could not be loaded into model."
                    )

        logger.info(f"Loaded checkpoint from {path}.")

    def _init_attached_module(
        self,
        cfg: AttachedModuleConfig,
        registry: Registry,
        storage: Mapping[str, Mapping[str, BaseAttachedModule]],
    ) -> tuple[str, str]:
        Module = registry.get(cfg.name)
        module_name = cfg.alias or cfg.name
        node_name = cfg.attached_to
        node: BaseNode = self.nodes[node_name]  # type: ignore
        if issubclass(Module, TorchMetricWrapper):
            if "task" not in cfg.params and self._core is not None:
                loader = self._core.loaders["train"]
                dataset = getattr(loader, "dataset", None)
                if isinstance(dataset, LuxonisDataset):
                    n_classes = len(dataset.get_classes()[1][node.task])
                    if n_classes == 1:
                        cfg.params["task"] = "binary"
                    else:
                        cfg.params["task"] = "multiclass"
                    logger.warning(
                        f"Parameter 'task' not specified for `TorchMetric` based '{module_name}' metric. "
                        f"Assuming task type based on the number of classes: {cfg.params['task']}. "
                        "If this is incorrect, please specify the 'task' parameter in the config."
                    )

        module = Module(**cfg.params, node=node)
        storage[node_name][module_name] = module  # type: ignore
        return module_name, node_name

    @staticmethod
    def _to_module_dict(
        modules: dict[str, dict[str, nn.Module]],
    ) -> nn.ModuleDict:
        return nn.ModuleDict(
            {
                node_name: nn.ModuleDict(node_modules)
                for node_name, node_modules in modules.items()
            }
        )

    @property
    def _progress_bar(self) -> BaseLuxonisProgressBar:
        return cast(
            BaseLuxonisProgressBar, self._trainer.progress_bar_callback
        )

    @rank_zero_only
    def _print_results(
        self, stage: str, loss: float, metrics: dict[str, dict[str, float]]
    ) -> None:
        """Prints validation metrics in the console."""

        logger.info(f"{stage} loss: {loss:.4f}")

        self._progress_bar.print_results(
            stage=stage, loss=loss, metrics=metrics
        )

        if self.main_metric is not None:
            main_metric_node, main_metric_name = self.main_metric.split("/")
            main_metric = metrics[main_metric_node][main_metric_name]
            logger.info(
                f"{stage} main metric ({self.main_metric}): {main_metric:.4f}"
            )

    def _average_losses(
        self, step_outputs: list[Mapping[str, Tensor | float | int]]
    ) -> dict[str, float]:
        avg_losses: dict[str, float] = defaultdict(float)

        for step_output in step_outputs:
            for key, value in step_output.items():
                avg_losses[key] += float(value)

        for key in avg_losses:
            avg_losses[key] /= len(step_outputs)
        return avg_losses
