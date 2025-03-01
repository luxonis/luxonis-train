from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger
from luxonis_ml.typing import PathType
from torch import Size, Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.attached_modules import ConfusionMatrix
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
    postprocess_metrics,
)


class LuxonisLightningModule(pl.LightningModule):
    logger: LuxonisTrackerPL

    _trainer: pl.Trainer

    __call__: Callable[..., LuxonisOutput]

    def __init__(
        self,
        cfg: Config,
        save_dir: PathType,
        input_shapes: dict[str, Size],
        dataset_metadata: DatasetMetadata | None = None,
        *,
        _core: "lxt.LuxonisModel | None" = None,
        **kwargs,
    ):
        """Lightning module for Luxonis models.

        Keeps track of the model's nodes, losses, metrics and
        visualizers,

        @type cfg: L{Config}
        @param cfg: The configuration.
        @type save_dir: str
        @param save_dir: Directory to which the model checkpoints will
            be saved.
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
        self._loss_accumulator = LossAccumulator()

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

    @property
    def tracker(self) -> LuxonisTrackerPL:
        """Returns the tracker.

        @type: L{LuxonisTrackerPL}
        """
        return self.logger

    @property
    def core(self) -> "lxt.LuxonisModel":
        """Returns the core model.

        @type: L{LuxonisModel}
        """
        if self._core is None:  # pragma: no cover
            raise ValueError("Core reference is not set.")
        return self._core

    @property
    def progress_bar(self) -> BaseLuxonisProgressBar:
        return cast(
            BaseLuxonisProgressBar, self._trainer.progress_bar_callback
        )

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

        processed: dict[str, Packet[Tensor]] = {}
        for node_name, node, input_names, unprocessed in self.nodes.traverse():
            if node.export and node.remove_on_export:
                continue

            input_names += self.nodes.inputs[node_name]

            node_inputs: list[Packet[Tensor]] = []
            for pred in input_names:
                if pred in processed:
                    node_inputs.append(processed[pred])
                else:
                    node_inputs.append({"features": [inputs[pred]]})

            outputs = node.run(node_inputs)

            processed[node_name] = outputs

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

            for computed_name in list(processed.keys()):
                if computed_name in self.outputs:
                    continue
                for unprocessed_name in unprocessed:
                    if computed_name in self.nodes.graph[unprocessed_name]:
                        break
                else:
                    del processed[computed_name]

        outputs_dict = {
            node_name: outputs
            for node_name, outputs in processed.items()
            if node_name in self.outputs
        }

        return LuxonisOutput(
            outputs=outputs_dict, losses=losses, visualizations=visualizations
        )

    @override
    def training_step(
        self, train_batch: tuple[dict[str, Tensor], Labels]
    ) -> Tensor:
        """Performs one step of training with provided batch."""
        outputs = self.forward(*train_batch)
        if not outputs.losses:
            raise ValueError(
                "Losses are empty, check if you have defined any loss"
            )

        loss, losses = compute_losses(
            self.cfg, outputs.losses, self.loss_weights, self.device
        )
        self._loss_accumulator.update(losses)
        return loss

    @override
    def validation_step(
        self, val_batch: tuple[dict[str, Tensor], Labels]
    ) -> dict[str, Tensor]:
        """Performs one step of validation with provided batch."""
        return self._evaluation_step("val", *val_batch)

    @override
    def test_step(
        self, test_batch: tuple[dict[str, Tensor], Labels]
    ) -> dict[str, Tensor]:
        """Performs one step of testing with provided batch."""
        return self._evaluation_step("test", *test_batch)

    @override
    def predict_step(
        self, batch: tuple[dict[str, Tensor], Labels]
    ) -> LuxonisOutput:
        """Performs one step of prediction with provided batch."""
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
        for module in self.modules():
            if isinstance(module, BaseNode):
                module.current_epoch = self.current_epoch

        self._loss_accumulator.clear()

    @override
    def on_train_epoch_end(self) -> None:
        for key, value in self._loss_accumulator.items():
            self.log(f"train/{key}", value, sync_dist=True)

    @override
    def on_validation_epoch_start(self) -> None:
        self._loss_accumulator.clear()
        self._n_logged_images = 0

    @override
    def on_validation_epoch_end(self) -> None:
        return self._evaluation_epoch_end("val")

    @override
    def on_test_epoch_start(self) -> None:
        self.on_validation_epoch_start()

    @override
    def on_test_epoch_end(self) -> None:
        return self._evaluation_epoch_end("test")

    # TODO: Parameter groups
    @override
    def configure_optimizers(
        self,
    ) -> tuple[list[Optimizer], list[LRScheduler]]:
        if self.training_strategy is not None:
            return self.training_strategy.configure_optimizers()
        return build_optimizers(self.cfg, self.parameters())

    @override
    def configure_callbacks(self) -> list[pl.Callback]:
        return build_callbacks(self.cfg, self.main_metric, self.save_dir)

    def set_export_mode(self, /, mode: bool) -> None:
        """Sets the export mode for the model."""
        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode(mode)

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
            for shapes in self.nodes.input_shapes.values()
            for input_name, shape in shapes.items()
        }
        input_names = list(inputs.keys())

        self.set_export_mode(True)

        outputs = self.forward(inputs).outputs
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

        # TODO: Remove
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

        old_forward = self.forward
        self.forward = export_forward  # type: ignore

        if "input_names" not in kwargs:
            assert set(input_names) == set(inputs.keys())
            kwargs["input_names"] = input_names
        if "output_names" not in kwargs:
            kwargs["output_names"] = output_names

        self.to_onnx(save_path, {"inputs": inputs}, **kwargs)

        self.forward = old_forward  # type: ignore

        self.set_export_mode(False)

        logger.info(f"Model exported to '{save_path}'")

        self.train()

        return output_names

    def load_checkpoint(self, path: PathType | None) -> None:
        """Loads checkpoint weights from provided path.

        Loads the checkpoints gracefully, ignoring keys that are not
        found in the model state dict or in the checkpoint.

        @type path: str | None
        @param path: Path to the checkpoint. If C{None}, no checkpoint
            will be loaded.
        """
        if path is None:
            return

        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True
        )

        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain `state_dict`.")

        state_dict = {}
        self_state_dict = self.state_dict()
        for key, value in checkpoint["state_dict"].items():
            if key not in self_state_dict.keys():
                logger.warning(
                    f"Key '{key}' from checkpoint not found in model state dict."
                )
            else:
                state_dict[key] = value

        for key in self_state_dict:
            if key not in state_dict.keys():
                logger.warning(f"Key '{key}' was not found in checkpoint.")
            else:
                try:
                    self_state_dict[key].copy_(state_dict[key])
                except Exception:
                    logger.warning(
                        f"Key '{key}' from checkpoint could not be loaded into model."
                    )

        logger.info(f"Loaded checkpoint from '{path}'.")

    def _evaluation_step(
        self,
        mode: Literal["test", "val"],
        inputs: dict[str, Tensor],
        labels: Labels,
    ) -> dict[str, Tensor]:
        input_image = inputs[self.image_source]

        if self._n_logged_images < self.cfg.trainer.n_log_images:
            images = get_denormalized_images(self.cfg, input_image)
        else:
            images = None

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
        self._loss_accumulator.update(losses)

        for node_name, visualizations in outputs.visualizations.items():
            for viz_name, viz_batch in visualizations.items():
                for viz in viz_batch:
                    name = f"{mode}/visualizations/{node_name}/{viz_name}"
                    if self._n_logged_images >= self.cfg.trainer.n_log_images:
                        continue
                    self.tracker.log_image(
                        f"{name}/{self._n_logged_images}",
                        viz.detach().cpu().numpy().transpose(1, 2, 0),
                        step=self.current_epoch,
                    )
                    self._n_logged_images += 1

        return losses

    def _evaluation_epoch_end(self, mode: Literal["test", "val"]) -> None:
        for name, value in self._loss_accumulator.items():
            self.log(f"{mode}/{name}", value, sync_dist=True)

        table = defaultdict(dict)

        for node_name, node_metrics in self.metrics.items():
            for metric_name, metric in node_metrics.items():
                values = postprocess_metrics(metric_name, metric.compute())
                metric.reset()

                for name, value in values.items():
                    if isinstance(metric, ConfusionMatrix):
                        self.tracker.log_matrix(
                            matrix=value.cpu().numpy(),
                            # TODO: Is the name correct?
                            name=f"{mode}/metrics/{self.current_epoch}/{name}",
                            step=self.current_epoch,
                        )
                    else:
                        table[node_name][metric_name] = value.cpu().item()
                        self.log(
                            f"{mode}/metric/{node_name}/{name}",
                            value,
                            sync_dist=True,
                        )

        self._print_results(
            stage="Validation" if mode == "val" else "Test",
            loss=self._loss_accumulator["loss"],
            metrics=table,
        )

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
