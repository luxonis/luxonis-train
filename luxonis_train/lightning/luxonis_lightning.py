import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from luxonis_ml import __version__ as luxonis_ml_version
from luxonis_ml.typing import PathType
from torch import Size, Tensor
from typing_extensions import override

import luxonis_train
from luxonis_train.attached_modules.visualizers import (
    combine_visualizations,
    get_denormalized_images,
)
from luxonis_train.callbacks import BaseLuxonisProgressBar
from luxonis_train.config import Config
from luxonis_train.nodes import BaseNode
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import DatasetMetadata, LuxonisTrackerPL

from .luxonis_output import LuxonisOutput
from .utils import (
    LossAccumulator,
    build_callbacks,
    build_losses,
    build_metrics,
    build_nodes,
    build_optimizers,
    build_training_strategy,
    build_visualizers,
    compute_losses,
    compute_visualization_buffer,
    get_model_execution_order,
    log_balanced_class_images,
    log_sequential_images,
    postprocess_metrics,
)


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
        save_dir: PathType,
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
        self._n_logged_images = 0
        self._class_log_counts: list[int] = []
        self._sequentially_logged_visualizations: list[
            dict[str, dict[str, Tensor]]
        ] = []
        self._needs_vis_buffering = True

        self._loss_accumulators = {
            "train": LossAccumulator(),
            "val": LossAccumulator(),
            "test": LossAccumulator(),
        }

        self.cfg = cfg
        self.image_source = cfg.loader.image_source
        self.dataset_metadata = dataset_metadata or DatasetMetadata()
        self.save_dir = Path(save_dir)
        self.outputs = self.cfg.model.outputs

        self.nodes = build_nodes(cfg, self.dataset_metadata, input_shapes)

        self.losses, self.loss_weights = build_losses(self.nodes, self.cfg)
        self.metrics, self.main_metric = build_metrics(self.nodes, self.cfg)
        self.visualizers = build_visualizers(self.nodes, self.cfg)
        self.training_strategy = build_training_strategy(self.cfg, self)

        self.load_checkpoint(self.cfg.model.weights)

        self.save_hyperparameters(
            {
                "luxonis_train_version": luxonis_train.__version__,
                "luxonis_ml_version": luxonis_ml_version,
            }
        )

    @property
    def progress_bar(self) -> BaseLuxonisProgressBar:
        return cast(
            BaseLuxonisProgressBar, self._trainer.progress_bar_callback
        )

    @property
    def tracker(self) -> LuxonisTrackerPL:
        return self.logger

    @property
    def core(self) -> "luxonis_train.core.LuxonisModel":
        """Returns the core model."""
        if self._core is None:  # pragma: no cover
            raise ValueError("Core reference is not set.")
        return self._core

    @override
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
        for node_name, node, input_names, unprocessed in self.nodes.traverse():
            if node.export and node.remove_on_export:
                continue
            input_names += self.nodes.inputs[node_name]

            node_inputs: list[Packet[Tensor]] = []
            for pred in input_names:
                if pred in computed:
                    node_inputs.append(computed[pred])
                else:
                    node_inputs.append({"features": [inputs[pred]]})

            outputs = node.run(node_inputs)

            computed[node_name] = outputs

            del node_inputs

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
                        visualizer.run(images, images, outputs, labels),
                    )
                    visualizations[node_name][viz_name] = viz

            for computed_name in list(computed.keys()):
                if computed_name in self.outputs:
                    continue
                for unprocessed_name in unprocessed:
                    if computed_name in self.nodes.graph[unprocessed_name]:
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

    def set_export_mode(self, *, mode: bool) -> None:
        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode(mode=mode)

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
        device_before = self.device

        self.eval()
        self.to("cpu")  # move to CPU to support deterministic .to_onnx()

        inputs = {
            input_name: torch.zeros([1, *shape]).to(self.device)
            for shapes in self.nodes.input_shapes.values()
            for input_name, shape in shapes.items()
        }

        inputs_deep_clone = {
            k: torch.zeros(elem.shape).to(self.device)
            for k, elem in inputs.items()
        }

        inputs_for_onnx = {"inputs": inputs_deep_clone}

        self.set_export_mode(mode=True)

        outputs = self.forward(inputs_deep_clone).outputs
        output_order = sorted(
            [
                (node_name, output_name, i)
                for node_name, outs in outputs.items()
                for output_name, out in outs.items()
                for i in range(len(out))
            ]
        )

        output_counts = defaultdict(int)
        for node_name, outs in outputs.items():
            output_counts[node_name] = sum(len(out) for out in outs.values())

        if self.cfg.exporter.output_names is not None:
            logger.warning(
                "The use of 'exporter.output_names' is deprecated and will be removed in a future version. "
                "If 'node.export_output_names' are provided, they will take precedence and overwrite 'exporter.output_names'. "
                "Please update your config to use 'node.export_output_names' directly."
            )

        export_output_names_used = False
        export_output_names_dict = {}
        for node_name, node in self.nodes.items():
            if node.export_output_names is not None:
                export_output_names_used = True
                if len(node.export_output_names) != output_counts[node_name]:
                    logger.warning(
                        f"Number of provided output names for node {node_name} "
                        f"({len(node.export_output_names)}) does not match "
                        f"number of outputs ({output_counts[node_name]}). "
                        f"Using default names."
                    )
                else:
                    export_output_names_dict[node_name] = (
                        node.export_output_names
                    )

        if (
            not export_output_names_used
            and self.cfg.exporter.output_names is not None
        ):
            len_names = len(self.cfg.exporter.output_names)
            if len_names != len(output_order):
                logger.warning(
                    f"Number of provided output names ({len_names}) does not match "
                    f"number of outputs ({len(output_order)}). Using default names."
                )
                self.cfg.exporter.output_names = None

            output_names = self.cfg.exporter.output_names or [
                f"{self.nodes.task_names[node_name]}/{node_name}/{output_name}/{i}"
                for node_name, output_name, i in output_order
            ]

            if not self.cfg.exporter.output_names:
                idx = 1
                # Set to output names required by DAI
                for i, output_name in enumerate(output_names):
                    if output_name.startswith("EfficientBBoxHead"):
                        output_names[i] = f"output{idx}_yolov6r2"
                        idx += 1
        else:
            output_names = []
            running_i = {}  # for case where export_output_names should be used but output node's output is split into multiple subnodes
            for node_name, output_name, i in output_order:
                if node_name in export_output_names_dict:
                    running_i[node_name] = (
                        running_i.get(node_name, -1) + 1
                    )  # if not present default to 0 otherwise add 1
                    output_names.append(
                        export_output_names_dict[node_name][
                            running_i[node_name]
                        ]
                    )
                else:
                    output_names.append(
                        f"{self.nodes.task_names[node_name]}/{node_name}/{output_name}/{i}"
                    )

        old_forward = self.forward

        def export_forward(inputs: dict[str, Tensor]) -> tuple[Tensor, ...]:
            old_outputs = old_forward(
                inputs,
                None,
                compute_loss=False,
                compute_metrics=False,
                compute_visualizations=False,
            ).outputs
            outputs = []
            for node_name, output_name, i in output_order:
                node_output = old_outputs[node_name][output_name]
                if isinstance(node_output, Tensor):
                    outputs.append(node_output)
                else:
                    outputs.append(node_output[i])
            return tuple(outputs)

        self.forward = export_forward  # type: ignore

        if "input_names" not in kwargs:
            kwargs["input_names"] = list(inputs.keys())
        if "output_names" not in kwargs:
            kwargs["output_names"] = output_names

        self.to_onnx(save_path, inputs_for_onnx, **kwargs)

        self.forward = old_forward  # type: ignore

        self.set_export_mode(mode=False)

        logger.info(f"Model exported to {save_path}")

        self.train()
        self.to(device_before)  # reset device after export

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
            formatted_node_name = self.nodes.formatted_name(node_name)
            for loss_name, loss_values in losses.items():
                if isinstance(loss_values, tuple):
                    loss, sublosses = loss_values
                else:
                    loss = loss_values
                    sublosses = {}

                loss *= self.loss_weights[loss_name]
                final_loss += loss
                training_step_output[
                    f"loss/{formatted_node_name}/{loss_name}"
                ] = loss.detach().cpu()
                if self.cfg.trainer.log_sub_losses and sublosses:
                    for subloss_name, subloss_value in sublosses.items():
                        training_step_output[
                            f"loss/{formatted_node_name}/{loss_name}/{subloss_name}"
                        ] = subloss_value.detach().cpu()
        training_step_output["loss"] = final_loss.detach().cpu()
        return final_loss, training_step_output

    @override
    def training_step(
        self, train_batch: tuple[dict[str, Tensor], Labels]
    ) -> Tensor:
        outputs = self.forward(*train_batch)
        if not outputs.losses:
            raise ValueError("Losses are empty, check if you defined any loss")

        loss, losses = compute_losses(
            self.cfg, outputs.losses, self.loss_weights, self.device
        )
        self._loss_accumulators["train"].update(losses)
        return loss

    @override
    def validation_step(
        self, val_batch: tuple[dict[str, Tensor], Labels]
    ) -> dict[str, Tensor]:
        return self._evaluation_step("val", *val_batch)

    @override
    def test_step(
        self, test_batch: tuple[dict[str, Tensor], Labels]
    ) -> dict[str, Tensor]:
        return self._evaluation_step("test", *test_batch)

    @override
    def predict_step(
        self, batch: tuple[dict[str, Tensor], Labels]
    ) -> LuxonisOutput:
        inputs, labels = batch
        images = get_denormalized_images(self.cfg, inputs[self.image_source])
        outputs = self.forward(
            inputs,
            labels,
            images=images,
            compute_visualizations=True,
            compute_loss=False,
            compute_metrics=False,
        )
        return outputs

    @override
    def on_train_epoch_start(self) -> None:
        for node in self.nodes.values():
            node.current_epoch = self.current_epoch

    @override
    def on_train_epoch_end(self) -> None:
        for name, value in self._loss_accumulators["train"].items():
            formated_name = (
                name.replace(
                    name.split("/")[1],
                    self.nodes.formatted_name(name.split("/")[1]),
                )
                if "/" in name
                else name
            )
            self.log(f"train/{formated_name}", value, sync_dist=True)
        self._loss_accumulators["train"].clear()

    @override
    def on_validation_epoch_end(self) -> None:
        return self._evaluation_epoch_end("val")

    @override
    def on_test_epoch_end(self) -> None:
        return self._evaluation_epoch_end("test")

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        pattern = re.compile(r"^(metrics|visualizers|losses)\..*_node\..*")
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not pattern.match(k)
        }
        checkpoint["execution_order"] = get_model_execution_order(self)
        checkpoint["config"] = self.cfg.model_dump(mode="json")
        checkpoint["dataset_metadata"] = self.dataset_metadata.dump()

    @override
    def configure_callbacks(self) -> list[pl.Callback]:
        return build_callbacks(
            self.cfg, self.main_metric, self.save_dir, self.nodes
        )

    @override
    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer],
        list[torch.optim.lr_scheduler.LRScheduler | dict[str, Any]],
    ]:
        if self.training_strategy is not None:
            return self.training_strategy.configure_optimizers()
        return build_optimizers(
            self.cfg, self.parameters(), self.main_metric, self.nodes
        )

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

        checkpoint = torch.load(  # nosemgrep
            path, map_location=self.device
        )

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
            if key not in state_dict:
                logger.warning(f"Key `{key}` was not found in checkpoint.")
            else:
                try:
                    self_state_dict[key].copy_(state_dict[key])
                except Exception:
                    logger.warning(
                        f"Key `{key}` from checkpoint could not be loaded into model."
                    )

        logger.info(f"Loaded checkpoint from {path}.")

    def _evaluation_step(
        self,
        mode: Literal["test", "val"],
        inputs: dict[str, Tensor],
        labels: Labels,
    ) -> dict[str, Tensor]:
        max_log_images = self.cfg.trainer.n_log_images
        input_image = inputs[self.image_source]

        # Smart logging is decided based on the classification task keys that are merged for all tasks
        cls_task_keys: list[str] | None = [
            k for k in labels if "/classification" in k
        ] or None
        images = None
        if self._n_logged_images < max_log_images:
            images = get_denormalized_images(self.cfg, input_image)

        outputs = self.forward(
            inputs,
            labels,
            images=images,
            compute_metrics=True,
            compute_visualizations=True,
        )

        _, losses = compute_losses(
            self.cfg, outputs.losses, self.loss_weights, self.device
        )

        self._loss_accumulators[mode].update(losses)

        if outputs.visualizations:
            if cls_task_keys is not None:
                # Smart logging: balance class representation
                labels_copy = {k: v.clone() for k, v in labels.items()}
                # Remove background class from segmentation tasks
                for k in (k for k in labels_copy if "/segmentation" in k):
                    cls_key = f"{k[: -len('/segmentation')]}/classification"
                    labels_copy[cls_key] = (
                        labels_copy[cls_key][:, 1:]
                        if labels_copy[cls_key].shape[1] > 1
                        else labels_copy[cls_key]
                    )

                n_classes = sum(
                    labels_copy[task].shape[1] for task in cls_task_keys
                )
                if (
                    not self._class_log_counts
                    or len(self._class_log_counts) != n_classes
                ):
                    self._class_log_counts = [0] * n_classes

                self._n_logged_images, self._class_log_counts, logged_idxs = (
                    log_balanced_class_images(
                        self.tracker,
                        self.nodes,
                        outputs.visualizations,
                        labels_copy,
                        cls_task_keys,
                        self._class_log_counts,
                        self._n_logged_images,
                        max_log_images,
                        mode,
                        self.current_epoch,
                    )
                )
                if self._needs_vis_buffering:
                    extra = compute_visualization_buffer(
                        self._sequentially_logged_visualizations,
                        outputs.visualizations,
                        logged_idxs,
                        max_log_images,
                    )
                    if extra:
                        self._sequentially_logged_visualizations.append(extra)
            else:
                # just log first N images
                self._n_logged_images = log_sequential_images(
                    self.tracker,
                    self.nodes,
                    outputs.visualizations,
                    self._n_logged_images,
                    max_log_images,
                    mode,
                    self.current_epoch,
                )

        return losses

    def _evaluation_epoch_end(self, mode: Literal["test", "val"]) -> None:
        for name, value in self._loss_accumulators[mode].items():
            formated_name = (
                name.replace(
                    name.split("/")[1],
                    self.nodes.formatted_name(name.split("/")[1]),
                )
                if "/" in name
                else name
            )
            self.log(f"{mode}/{formated_name}", value, sync_dist=True)

        table = defaultdict(dict)
        for node_name, node_metrics in self.metrics.items():
            formatted_node_name = self.nodes.formatted_name(node_name)
            for metric_name, metric in node_metrics.items():
                values = postprocess_metrics(metric_name, metric.compute())
                metric.reset()

                for name, value in values.items():
                    if value.dim() == 2:
                        self.tracker.log_matrix(
                            matrix=value.cpu().numpy(),
                            name=f"{mode}/metrics/{self.current_epoch}/"
                            f"{formatted_node_name}/{name}",
                            step=self.current_epoch,
                        )
                    else:
                        table[node_name][name] = value.cpu().item()
                        self.log(
                            f"{mode}/metric/{formatted_node_name}/{name}",
                            value,
                            sync_dist=True,
                        )

        self._print_results(
            stage="Validation" if mode == "val" else "Test",
            loss=self._loss_accumulators[mode]["loss"],
            metrics=table,
        )

        if self._n_logged_images != self.cfg.trainer.n_log_images:
            logger.warning(
                f"Logged images ({self._n_logged_images}) != expected ({self.cfg.trainer.n_log_images}). Possible reasons: "
                f"class imbalance or a small number of images in the split. Trying to log more images."
            )
            for (
                missing_visualizations
            ) in self._sequentially_logged_visualizations:
                self._n_logged_images = log_sequential_images(
                    self.tracker,
                    self.nodes,
                    missing_visualizations,
                    self._n_logged_images,
                    self.cfg.trainer.n_log_images,
                    mode,
                    self.current_epoch,
                )
        else:
            self._needs_vis_buffering = False

        self._sequentially_logged_visualizations.clear()

        self._n_logged_images = 0
        if self._class_log_counts:
            self._class_log_counts = [0] * len(self._class_log_counts)
        self._loss_accumulators[mode].clear()

    @rank_zero_only
    def _print_results(
        self, stage: str, loss: float, metrics: dict[str, dict[str, float]]
    ) -> None:
        """Prints validation metrics in the console."""
        logger.info(f"{stage} loss: {loss:.4f}")

        self.progress_bar.print_results(
            stage=stage, loss=loss, metrics=metrics
        )

        if self.main_metric is not None:
            node_name, metric_name = self.main_metric

            value = metrics[node_name][metric_name]
            logger.info(
                f"{stage} main metric ({node_name}/{metric_name}): {value:.4f}"
            )

    def get_mlflow_logging_keys(self) -> dict[str, list[str]]:
        """
        Returns a dictionary with two lists of keys:
        1) "metrics"    -> Keys expected to be logged as standard metrics
        2) "artifacts"  -> Keys expected to be logged as artifacts (e.g. confusion_matrix.json, visualizations)
        """
        artifact_keys = set()
        metric_keys = set()

        val_eval_epochs = []
        for i in range(
            self.cfg.trainer.validation_interval,
            self.cfg.trainer.epochs + 1,
            self.cfg.trainer.validation_interval,
        ):
            val_eval_epochs.append(max(0, i - 1))
        test_eval_epoch = self.cfg.trainer.epochs

        for mode in ["train", "val", "test"]:
            metric_keys.add(f"{mode}/loss")
            for node_name, node_losses in self.losses.items():
                formatted_node_name = self.nodes.formatted_name(node_name)
                for loss_name in node_losses.keys():
                    metric_keys.add(
                        f"{mode}/loss/{formatted_node_name}/{loss_name}"
                    )

        for node_name, node_metrics in self.metrics.items():
            formatted_node_name = self.nodes.formatted_name(node_name)
            for metric_name, metric in node_metrics.items():
                values = postprocess_metrics(metric_name, metric.compute())
                for sub_name in values.keys():
                    if "confusion_matrix" in sub_name:
                        for epoch_idx in sorted([0] + val_eval_epochs):
                            artifact_keys.add(
                                f"val/metrics/{epoch_idx}/{formatted_node_name}/confusion_matrix.json"
                            )
                        artifact_keys.add(
                            f"test/metrics/{test_eval_epoch}/{formatted_node_name}/confusion_matrix.json"
                        )
                    else:
                        for epoch_idx in sorted(val_eval_epochs):
                            metric_keys.add(
                                f"val/metric/{formatted_node_name}/{sub_name}"
                            )
                        metric_keys.add(
                            f"test/metric/{formatted_node_name}/{sub_name}"
                        )

        for node_name, visualizations in self.visualizers.items():
            formatted_node_name = self.nodes.formatted_name(node_name)
            for viz_name in visualizations.keys():
                for epoch_idx in sorted([0] + val_eval_epochs):
                    for i in range(self.cfg.trainer.n_log_images):
                        artifact_keys.add(
                            f"val/visualizations/{formatted_node_name}/{viz_name}/{epoch_idx}/{i}.png"
                        )
                for i in range(self.cfg.trainer.n_log_images):
                    artifact_keys.add(
                        f"test/visualizations/{formatted_node_name}/{viz_name}/{test_eval_epoch}/{i}.png"
                    )
        for callback in self.cfg.trainer.callbacks:
            if callback.name == "UploadCheckpoint":
                artifact_keys.update(
                    {"best_val_metric.ckpt", "min_val_loss.ckpt"}
                )
            elif callback.name == "ExportOnTrainEnd":
                artifact_keys.add(
                    f"{self.cfg.exporter.name or self.cfg.model.name}.onnx"
                )
            elif callback.name == "ArchiveOnTrainEnd":
                artifact_keys.add(
                    f"{self.cfg.exporter.name or self.cfg.model.name}.onnx.tar.xz"
                )

        artifact_keys.update(
            {
                "luxonis_train.log",
                "training_config.yaml",
                f"{self.cfg.model.name}.yaml",
            }
        )

        return {
            "metrics": sorted(metric_keys),
            "artifacts": sorted(artifact_keys),
        }
