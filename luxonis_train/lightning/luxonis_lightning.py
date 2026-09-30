"""The Lightning module that runs the node graph.

It builds the nodes, runs them in topological order, computes the losses
and the metrics, and decides what to log on each epoch.

"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from loguru import logger
from luxonis_ml import __version__ as luxonis_ml_version
from luxonis_ml.typing import PathType
from packaging import version
from semver import Version
from torch import Size, Tensor
from torch.nn.modules.module import _IncompatibleKeys
from torch.optim import Optimizer
from typing_extensions import Self, override

import luxonis_train
from luxonis_train.attached_modules.visualizers import (
    combine_visualizations,
    get_denormalized_images,
)
from luxonis_train.callbacks import BaseLuxonisProgressBar
from luxonis_train.config import Config
from luxonis_train.lightning.training_plan import (
    TrainingPlanRuntime,
    build_training_plan,
    resolve_training_plan,
)
from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.blocks.reparameterizable import Reparameterizable
from luxonis_train.registry import _INTERNAL
from luxonis_train.typing import Labels, Packet
from luxonis_train.utils import DatasetMetadata, LuxonisTrackerPL
from luxonis_train.utils.checkpoint import (
    CHECKPOINT_FILTERED_STATE_DICT_PATTERN,
    filter_checkpoint_state_dict,
)

from .luxonis_output import LuxonisOutput
from .utils import (
    LossAccumulator,
    Nodes,
    NodeWrapper,
    build_training_strategy,
    check_tensor_device,
    compute_losses,
    compute_visualization_buffer,
    get_model_execution_order,
    log_balanced_class_images,
    log_metric_artifacts,
    log_sequential_images,
    metric_artifact_image_name,
    mlflow_image_key,
    postprocess_metrics,
)

_TRAINING_PROGRESS_METRIC_KEYS = {
    f"{mode}/{suffix}"
    for mode in ("train", "val", "test")
    for suffix in (
        "batch_total_sec",
        "epoch_progress_percent",
        "epoch_duration_sec",
        "epoch_completion_sec",
    )
}

_NodeLosses = dict[str, dict[str, Tensor | tuple[Tensor, dict[str, Tensor]]]]


class LuxonisLightningModule(pl.LightningModule):
    """The Lightning module that holds the whole model.

    The module builds every node of the config and keeps them in
    ``nodes``, together with the losses, the metrics, and the
    visualizers attached to each node. The model topology is an
    acyclic graph of nodes. ``nodes.graph`` stores it as a mapping
    from a node name to the names of the nodes that feed it.

    `full_forward` runs the graph. Lightning drives training,
    validation, testing, and prediction through the ``*_step``
    methods and the ``on_*`` hooks. `LuxonisModel` owns the module
    and attaches it to a trainer.

    Attributes:
        cfg (Config): The config the module was built from.
        image_source (str): The name of the loader input that holds
            the image, from ``cfg.loader.image_source``.
        dataset_metadata (DatasetMetadata): The metadata of the
            dataset. An empty `DatasetMetadata` when the constructor
            got none.
        save_dir (``Path``): The directory where the checkpoints and
            the logs go.
        outputs (list[str]): The names of the output nodes, from
            ``cfg.model.outputs``.
        nodes (Nodes): The node wrappers, keyed by node name.
        training_strategy (BaseTrainingStrategy | None): The strategy
            that ``cfg.trainer.training_strategy`` names, or ``None``
            when the config names none. A strategy that predates the
            rule-based API comes wrapped in a `LegacyStrategyAdapter`.

    """

    _trainer: pl.Trainer
    logger: LuxonisTrackerPL

    _ckpt_predefined_model: dict[str, Any] | None = None
    """The predefined-model pin inherited from a checkpoint.

    `LuxonisModel` sets it only when the config itself comes from that
    checkpoint. A load of weights into a user-supplied config leaves it
    ``None``. `on_save_checkpoint` writes it into the checkpoint when
    the config names no predefined model.

    """

    __call__: Callable[..., tuple[Tensor, ...]]

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
        """Build the module from a config.

        The constructor does the following steps:

        - It builds the nodes and their attached modules.
        - It builds the training strategy.
        - It loads the checkpoint that ``cfg.model.weights`` names, when
          there is one. That load leaves the module in evaluation mode.
        - It saves the versions of ``luxonis_train`` and ``luxonis_ml``
          as hyperparameters.

        Args:
            cfg (Config): The config that defines the model, the
                loader, and the trainer.
            save_dir (``PathType``): The directory where the
                checkpoints and the logs go, as a string or a
                `pathlib.Path`.
            input_shapes (``dict[str, Size]``): The shape of every
                loader input, keyed by input name and without the
                batch dimension.
            dataset_metadata (DatasetMetadata | None): The metadata of
                the dataset. ``None`` builds an empty
                `DatasetMetadata`.
            _core (LuxonisModel | None): The `LuxonisModel` that owns
                this module. `core` raises without it.
            **kwargs (``Any``): Extra keyword arguments for the
                ``LightningModule`` constructor.

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

        self.nodes = Nodes(cfg, self.dataset_metadata, input_shapes)

        self.training_strategy = build_training_strategy(self.cfg, self)

        self.load_checkpoint(self.cfg.model.weights)

        self.save_hyperparameters(
            {
                "luxonis_train_version": luxonis_train.__version__,
                "luxonis_ml_version": luxonis_ml_version,
            }
        )
        self._restore_validation_interval_after_first_epoch = False
        self._original_check_val_every_n_epoch: int | None = None
        self._training_plan: TrainingPlanRuntime | None = None

    @override
    def load_state_dict(
        self, state_dict: Mapping[str, Tensor], strict: bool = True
    ) -> _IncompatibleKeys:
        """Load a state dict, and relax the check when a run resumes.

        Lightning calls this method when it restores a run from a
        checkpoint, with ``strict`` set to the ``strict_loading`` of
        the module, ``True`` by default. When
        ``cfg.trainer.resume_training`` is off, the method behaves
        like `torch.nn.Module.load_state_dict`.

        When ``cfg.trainer.resume_training`` is on:

        - With ``cfg.trainer.strict_weights_loading`` off, the method
          loads the state dict with ``strict=False`` and returns every
          mismatch.
        - With ``cfg.trainer.strict_weights_loading`` on, the method
          leaves out the keys that point from a loss, a metric, or a
          visualizer back at its node. Such a key starts with
          ``nodes.<node>.losses.``, ``nodes.<node>.metrics.``, or
          ``nodes.<node>.visualizers.`` and holds ``_node.``. The
          method ignores the same keys in the result. It raises for
          any other missing or unexpected key.

        Args:
            state_dict (``Mapping[str, Tensor]``): The parameters and
                the buffers to load, keyed by their name in this
                module.
            strict (bool): Whether every key must match. Ignored when
                a run resumes.

        Returns:
            ``_IncompatibleKeys``: The ``missing_keys`` and the
            ``unexpected_keys`` of the load. Both are empty after a
            strict resume.

        Raises:
            RuntimeError: When a run resumes with strict weight
                loading and a key that is not such an attached-module
                key is missing or unexpected.

        """
        if self.cfg.trainer.resume_training:
            filtered_state_dict = (
                filter_checkpoint_state_dict(state_dict)
                if self.cfg.trainer.strict_weights_loading
                else state_dict
            )
            incompatible = super().load_state_dict(
                filtered_state_dict, strict=False
            )
            if not self.cfg.trainer.strict_weights_loading:
                return incompatible

            missing_keys = [
                key
                for key in incompatible.missing_keys
                if not CHECKPOINT_FILTERED_STATE_DICT_PATTERN.match(key)
            ]
            unexpected_keys = [
                key
                for key in incompatible.unexpected_keys
                if not CHECKPOINT_FILTERED_STATE_DICT_PATTERN.match(key)
            ]

            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Error(s) in loading state_dict for "
                    f"{self.__class__.__name__}:\n"
                    + (
                        f"\tMissing key(s): {', '.join(missing_keys)}.\n"
                        if missing_keys
                        else ""
                    )
                    + (
                        f"\tUnexpected key(s): {', '.join(unexpected_keys)}.\n"
                        if unexpected_keys
                        else ""
                    )
                )
            return _IncompatibleKeys([], [])
        return super().load_state_dict(state_dict, strict=strict)

    @property
    def progress_bar(self) -> BaseLuxonisProgressBar:
        """The progress bar callback of the attached trainer.

        `LuxonisModel` registers a `LuxonisRichProgressBar` or a
        `LuxonisTQDMProgressBar`, and the module prints the results of
        an evaluation epoch through it. Without an attached trainer, the
        lookup raises ``AttributeError``.

        """
        return cast(
            BaseLuxonisProgressBar, self._trainer.progress_bar_callback
        )

    @property
    def tracker(self) -> LuxonisTrackerPL:
        """The logger of the attached trainer, as a `LuxonisTrackerPL`.

        `LuxonisModel` creates the trainer with a `LuxonisTrackerPL` as
        its logger. The value is ``None`` while no trainer is attached.

        """
        return self.logger

    @property
    def core(self) -> "luxonis_train.core.LuxonisModel":
        """The `LuxonisModel` that owns this module.

        Raises:
            ValueError: When the module was built without ``_core``.

        """
        if self._core is None:  # pragma: no cover
            raise ValueError("Core reference is not set.")
        return self._core

    @override
    def forward(
        self, inputs: dict[str, Tensor] | Tensor
    ) -> tuple[Tensor, ...]:
        """Run the graph and return the outputs as a flat tuple.

        This is the entry point of a direct call of the module and of
        the ONNX export. It runs `full_forward` without the losses,
        the metrics, and the visualizations, and it flattens the
        packets of the output nodes.

        The tuple is sorted by node name, then by output key, then by
        index. A list value contributes each of its tensors. A single
        tensor value repeats once for every entry of its first
        dimension. The export uses a batch of one.

        Args:
            inputs (``dict[str, Tensor] | Tensor``): The loader inputs,
                keyed by input name. A bare tensor is the input named
                ``image_source``.

        Returns:
            ``tuple[Tensor, ...]``: The flattened outputs of the output
            nodes.

        """
        outputs = self.full_forward(
            inputs,
            compute_loss=False,
            compute_metrics=False,
            compute_visualizations=False,
        ).outputs

        output_order = _output_order(outputs)
        new_outputs = []
        for node_name, output_name, i in output_order:
            node_output = outputs[node_name][output_name]
            if isinstance(node_output, Tensor):
                new_outputs.append(node_output)
            else:
                new_outputs.append(node_output[i])

        return tuple(new_outputs)

    def full_forward(
        self,
        inputs: dict[str, Tensor] | Tensor,
        labels: Labels | None = None,
        images: Tensor | None = None,
        *,
        compute_loss: bool = True,
        compute_metrics: bool = False,
        compute_visualizations: bool = False,
    ) -> LuxonisOutput:
        """Run every node of the graph and collect its results.

        The nodes run in topological order. A node runs after every
        node that feeds it. The method skips a node whose ``export``
        flag and ``remove_on_export`` are both set. The method drops
        the output of a node from memory once no later node reads it,
        unless the node is an output node.

        After each node runs, its attached modules move to the device
        of the module and run:

        - With ``compute_loss`` on and ``labels`` given, every loss of
          the node runs on the outputs and the labels. While the module
          is in training mode, the method puts the loss in training
          mode first.
        - With ``compute_metrics`` on and ``labels`` given, every
          metric of the node updates its state. The method does not
          return the metric values.
        - With ``compute_visualizations`` on and ``images`` given,
          every visualizer of the node draws on ``images``, and
          `combine_visualizations` merges its result into one image
          batch.

        Args:
            inputs (``dict[str, Tensor] | Tensor``): The loader inputs,
                keyed by input name. A bare tensor is the input named
                ``image_source``. The tensors move to the device of
                the module.
            labels (Labels | None): The labels of the batch, keyed as
                ``<task>/<label>``. The tensors move to the device of
                the module. ``None`` skips the losses and the metrics.
            images (``Tensor | None``): The denormalized images of the
                batch, of shape ``[B, C, H, W]``, that the visualizers
                draw on. ``None`` skips the visualizations.
            compute_loss (bool): Whether to run the losses.
            compute_metrics (bool): Whether to update the metrics.
            compute_visualizations (bool): Whether to run the
                visualizers.

        Returns:
            LuxonisOutput: ``outputs`` holds the packet of every
            output node that ran, keyed by node name. ``losses`` holds
            the value of every loss, keyed by node name and loss name;
            a value is a tensor, or a tuple of the tensor and its
            sub-losses.
            ``visualizations`` holds the image batch of every
            visualizer, of shape ``[B, C, H, W]``, keyed by node name
            and visualizer name. ``metrics`` stays empty.

        """
        if isinstance(inputs, Tensor):
            inputs = {self.image_source: inputs}

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if labels is not None:
            labels = {k: v.to(self.device) for k, v in labels.items()}
        losses: _NodeLosses = defaultdict(dict)
        visualizations: dict[str, dict[str, Tensor]] = defaultdict(dict)

        computed: dict[str, Packet[Tensor]] = {}
        for node_name, node, _, unprocessed in self.nodes.traverse():
            if node.module.export and node.module.remove_on_export:
                continue
            node_inputs = _node_inputs(node.inputs, computed, inputs)
            outputs = node.module.run(node_inputs)
            computed[node_name] = outputs
            del node_inputs

            self._collect_node_results(
                node,
                node_name,
                outputs,
                labels,
                images,
                losses,
                visualizations,
                compute_loss=compute_loss,
                compute_metrics=compute_metrics,
                compute_visualizations=compute_visualizations,
            )
            self._drop_unused_outputs(computed, unprocessed)

        outputs_dict = {
            node_name: outputs
            for node_name, outputs in computed.items()
            if node_name in self.outputs
        }

        return LuxonisOutput(
            outputs=outputs_dict, losses=losses, visualizations=visualizations
        )

    @override
    def train(self, mode: bool = True) -> Self:
        """Set the training mode of the module and of every node.

        Besides the recursion of `torch.nn.Module.train`, the method
        calls ``train`` on every `NodeWrapper`. The wrapper passes the
        mode to the node and to the losses, the metrics, and the
        visualizers attached to it.

        Args:
            mode (bool): ``True`` for training mode, ``False`` for
                evaluation mode.

        Returns:
            ``Self``: The module.

        """
        super().train(mode)
        for node in self.nodes.values():
            node.train(mode)
        return self

    def set_export_mode(self, mode: bool) -> Self:
        """Switch every node into or out of export mode.

        The method calls `BaseNode.set_export_mode` on every `BaseNode`
        among the submodules of the module, nested ones included. A
        node reads its export flag to change its outputs, and the call
        reparameterizes its `Reparameterizable` blocks. With ``False``,
        the call restores those blocks.

        Args:
            mode (bool): ``True`` to enter export mode, ``False`` to
                leave it.

        Returns:
            ``Self``: The module.

        """
        for module in self.modules():
            if isinstance(module, BaseNode):
                module.set_export_mode(mode=mode)
        return self

    def reparameterize(self) -> Self:
        """Reparameterize every `Reparameterizable` block of the model.

        Unlike `set_export_mode`, the method leaves the export mode of
        the nodes unchanged. `set_export_mode` with ``False`` restores
        the blocks. `LuxonisModel.quantize` calls it before the
        quantization.

        Returns:
            ``Self``: The module.

        """
        for module in self.modules():
            if isinstance(module, Reparameterizable):
                module.reparameterize()
        return self

    def export_onnx(self, save_path: PathType, **kwargs) -> Path:
        """Export the model to ONNX.

        The method puts the module in evaluation mode, moves it to the
        CPU, and enters export mode. It builds a zero input of shape
        ``[1, *shape]`` for every loader input that a node reads. It
        runs the graph once to name the outputs, and calls ``to_onnx``
        of the Lightning module. It then logs the path, leaves export
        mode, puts the module in **training mode**, and moves it back
        to its device.

        The default ``input_names`` are the loader input names. The
        default ``output_names`` are the ``export_output_names`` of a
        node when their count matches its outputs, and
        ``<task>/<node>/<output>/<index>`` otherwise. A count mismatch
        logs a warning. On PyTorch 2.5 and later, ``dynamo`` defaults
        to ``False``.

        Args:
            save_path (``PathType``): The path of the ONNX file, as a
                string or a `pathlib.Path`.
            **kwargs (``Any``): Extra keyword arguments for
                `torch.onnx.export`, such as ``opset_version`` and
                ``dynamic_axes``.

        Returns:
            ``Path``: ``save_path`` as a `pathlib.Path`.

        """
        device = self.device

        self.eval()
        self.to("cpu")  # move to CPU to support deterministic .to_onnx()

        inputs = {
            input_name: torch.zeros([1, *shape]).to(self.device)
            for shapes in self.nodes.loader_input_shapes.values()
            for input_name, shape in shapes.items()
        }
        if "input_names" not in kwargs:
            kwargs["input_names"] = list(inputs.keys())

        self.set_export_mode(mode=True)

        output_names = self._get_output_onnx_names(deepcopy(inputs))

        if "output_names" not in kwargs:
            kwargs["output_names"] = output_names

        if version.parse(torch.__version__) >= version.parse("2.5.0"):
            # PyTorch 2.9 introduces a breaking change that
            # sets the default value to True
            kwargs.setdefault("dynamo", False)

        self.to_onnx(save_path, {"inputs": inputs}, **kwargs)

        logger.info(f"Model exported to {save_path}")

        self.set_export_mode(mode=False)
        self.train()
        self.to(device)  # reset device after export

        return Path(save_path)

    def compute_training_loss(
        self, train_batch: tuple[dict[str, Tensor] | Tensor, Labels]
    ) -> Tensor:
        """Run one training batch and return its total loss.

        The method runs `full_forward` with the losses on, sums them
        with `compute_losses`, and records every loss value in the
        training loss accumulator that `on_train_epoch_end` logs.

        Args:
            train_batch (``tuple[dict[str, Tensor] | Tensor, Labels]``):
                The loader inputs and the labels of the batch.

        Returns:
            ``Tensor``: The sum of every loss, each already scaled by
            its config ``weight``, of shape ``[1]`` on the device of
            the module.

        Raises:
            ValueError: When no node produced a loss.

        """
        outputs = self.full_forward(*train_batch)
        if not outputs.losses:
            raise ValueError("Losses are empty, check if you defined any loss")

        loss, losses = compute_losses(self.cfg, outputs.losses, self.device)
        self._loss_accumulators["train"].update(losses)
        return loss

    @override
    def training_step(
        self, train_batch: tuple[dict[str, Tensor] | Tensor, Labels]
    ) -> Tensor:
        """Run one batch of the training loop.

        Lightning calls it once for every batch of the training loader
        and backpropagates the returned loss.

        Args:
            train_batch (``tuple[dict[str, Tensor] | Tensor, Labels]``):
                The loader inputs and the labels of the batch.

        Returns:
            ``Tensor``: The total loss from `compute_training_loss`.

        """
        return self.compute_training_loss(train_batch)

    @override
    def validation_step(
        self, val_batch: tuple[dict[str, Tensor] | Tensor, Labels]
    ) -> dict[str, Tensor]:
        """Run one batch of the validation loop.

        Lightning calls it once for every batch of the validation
        loader. The method runs `full_forward` with the losses, the
        metrics, and the visualizers on. It records the loss values in
        the validation loss accumulator. It runs the visualizers only
        while the epoch has logged fewer images than
        ``cfg.trainer.n_log_images``. It logs their images to the
        tracker. When a label key contains ``/classification``, it
        balances the logged images across the classes and buffers the
        skipped images for `on_validation_epoch_end`. Otherwise it logs
        the first images of the epoch.

        Args:
            val_batch (``tuple[dict[str, Tensor] | Tensor, Labels]``):
                The loader inputs and the labels of the batch.

        Returns:
            ``dict[str, Tensor]``: The loss values of the batch on the
            CPU, keyed ``"loss"`` for the total,
            ``"loss/<node>/<loss>"`` for each loss, and
            ``"loss/<node>/<loss>/<sub>"`` for each sub-loss when
            ``cfg.trainer.log_sub_losses`` is on.

        """
        return self._evaluation_step("val", *val_batch)

    @override
    def test_step(
        self, test_batch: tuple[dict[str, Tensor] | Tensor, Labels]
    ) -> dict[str, Tensor]:
        """Run one batch of the test loop.

        Lightning calls it once for every batch of the test loader.
        The method does the same as `validation_step` with the
        ``test`` loss accumulator and the ``test`` image prefix.

        Args:
            test_batch (``tuple[dict[str, Tensor] | Tensor, Labels]``):
                The loader inputs and the labels of the batch.

        Returns:
            ``dict[str, Tensor]``: The loss values of the batch on the
            CPU, keyed as in `validation_step`.

        """
        return self._evaluation_step("test", *test_batch)

    @override
    def predict_step(
        self, batch: tuple[dict[str, Tensor] | Tensor, Labels]
    ) -> LuxonisOutput:
        """Run one batch of the prediction loop.

        Lightning calls it once for every batch of the prediction
        loader. The method denormalizes the image input into ``uint8``
        images. It then runs `full_forward` on the batch with those
        images, the visualizers on, and the losses and the metrics off.

        Args:
            batch (``tuple[dict[str, Tensor] | Tensor, Labels]``): The
                loader inputs and the labels of the batch.

        Returns:
            LuxonisOutput: The packet of every output node and the
            image of every visualizer. ``losses`` and ``metrics`` are
            empty.

        """
        inputs, labels = batch
        images = get_denormalized_images(
            self.cfg,
            inputs[self.image_source] if isinstance(inputs, dict) else inputs,
        )
        return self.full_forward(
            inputs,
            labels,
            images=images,
            compute_visualizations=True,
            compute_loss=False,
            compute_metrics=False,
        )

    @override
    def setup(self, stage: str) -> None:
        """Make the first epoch validate when the config asks for it.

        Lightning calls it at the start of every stage. The method acts
        only when all of these hold:

        - The stage is ``fit``.
        - ``cfg.trainer.run_validation_after_first_epoch`` is on.
        - The current epoch is 0.
        - ``trainer.check_val_every_n_epoch`` is set and above 1.

        It acts once per run.

        Lightning decides at the end of an epoch whether validation
        runs from ``trainer.check_val_every_n_epoch``. The method
        stores that value and sets it to ``1``, so the first epoch
        validates even when ``validation_interval`` would skip it.
        `on_validation_epoch_end` restores the stored value after that
        first validation, so the rest of the run follows the
        configured interval.

        Args:
            stage (str): The stage that starts: ``"fit"``,
                ``"validate"``, ``"test"``, or ``"predict"``.

        """
        if getattr(stage, "value", stage) != "fit":
            return

        if (
            not self.cfg.trainer.run_validation_after_first_epoch
            or self.trainer.current_epoch != 0
            or self.trainer.check_val_every_n_epoch is None
            or self.trainer.check_val_every_n_epoch <= 1
        ):
            return

        if self._restore_validation_interval_after_first_epoch:
            return

        self._original_check_val_every_n_epoch = (
            self.trainer.check_val_every_n_epoch
        )
        self.trainer.check_val_every_n_epoch = 1
        self._restore_validation_interval_after_first_epoch = True

    @override
    def on_train_epoch_start(self) -> None:
        """Tell every node the number of the epoch that starts.

        Lightning calls it at the start of every training epoch. The
        method sets ``current_epoch`` on the module of every node, which
        the node and its attached modules read.

        """
        for node in self.nodes.values():
            node.module.current_epoch = self.current_epoch

    @override
    def on_train_epoch_end(self) -> None:
        """Log the mean training losses of the epoch.

        Lightning calls it at the end of every training epoch. The
        method logs every entry of the training loss accumulator as
        ``train/<name>`` with ``sync_dist=True``. It replaces the node
        name in ``<name>`` with the log name of the node, see
        `Nodes.formatted_name`. It then clears the accumulator.

        """
        _log_accumulated_losses(self, "train")
        self._loss_accumulators["train"].clear()

    @override
    def on_validation_epoch_end(self) -> None:
        """Log the validation results of the epoch.

        Lightning calls it at the end of every validation epoch, the
        sanity check included. The method does the following steps:

        - It logs the mean validation losses as ``val/<name>``, as
          `on_train_epoch_end` does for training.
        - It computes every metric. On the main process, outside the
          sanity check, it uploads the image artifacts of the metric to
          the tracker. It then resets the metric. It logs a scalar
          value as ``val/metric/<task>-<node>/<name>`` with
          ``sync_dist=True``. A 2-D value goes to the tracker as a
          matrix named ``val/metrics/<epoch>/<task>-<node>/<name>``,
          with the row labels of
          `BaseLuxonisProgressBar.format_matrix_for_printing` as its
          class names. ``<task>-<node>`` is the log name of the node,
          see `Nodes.formatted_name`. ``<name>`` is the metric
          identifier, or the name of a sub-metric when
          ``cfg.trainer.log_sub_metrics`` is on.
        - On the main process, it logs the loss, prints the metrics
          through the progress bar, and logs the main metric value.
        - When the epoch logged fewer than ``cfg.trainer.n_log_images``
          images, it logs a warning. It then logs the visualizations
          that the epoch buffered. When the count matches, it stops
          the buffering of skipped images for the rest of the run.
        - It clears the buffered visualizations, the image counters,
          and the loss accumulator.

        After the validation of epoch 0, outside the sanity check, it
        puts back the ``trainer.check_val_every_n_epoch`` that `setup`
        replaced.

        Raises:
            RuntimeError: When the run uses DDP and a metric value is
                not on the device of the module.

        """
        self._evaluation_epoch_end("val")

        if (
            not self.trainer.sanity_checking
            and self._restore_validation_interval_after_first_epoch
            and self.current_epoch == 0
            and self._original_check_val_every_n_epoch is not None
        ):
            self.trainer.check_val_every_n_epoch = (
                self._original_check_val_every_n_epoch
            )
            self._restore_validation_interval_after_first_epoch = False
            self._original_check_val_every_n_epoch = None

    @override
    def on_test_epoch_end(self) -> None:
        """Log the test results of the epoch.

        Lightning calls it at the end of the test epoch. The method
        does the same as `on_validation_epoch_end` with the ``test``
        prefix, without the change of the validation interval.

        Raises:
            RuntimeError: When the run uses DDP and a metric value is
                not on the device of the module.

        """
        return self._evaluation_epoch_end("test")

    @override
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Add the metadata of the run to a checkpoint.

        Lightning calls it before it writes a checkpoint. The method
        changes ``checkpoint`` in place:

        - It drops the keys of ``state_dict`` that point from a loss,
          a metric, or a visualizer back at its node.
        - It adds ``version``, the version of ``luxonis_train``.
        - It adds ``execution_order``, the names of the leaf modules
          that hold parameters, in the order a forward pass runs them.
        - It adds ``config``, the dump of the config.
        - It adds ``dataset_metadata``, the dump of the dataset
          metadata.
        - It adds ``predefined_model``: the one of the config, with
          ``latest`` resolved to a version number, or else the pin
          inherited from the loaded checkpoint. Without either, the
          method removes the key.

        The execution order comes from a forward pass on zero inputs,
        which leaves the module in evaluation mode.

        Args:
            checkpoint (``dict[str, Any]``): The checkpoint dictionary
                that Lightning is about to write.

        """
        super().on_save_checkpoint(checkpoint)
        self._add_custom_data_to_checkpoint(checkpoint)

    @override
    def configure_callbacks(self) -> list[pl.Callback]:
        """Build the callbacks that every run gets.

        Lightning calls it at the start of every ``fit``, ``validate``,
        ``test``, or ``predict`` call of the trainer. The method
        returns `Nodes.build_callbacks` of ``save_dir``, which holds,
        in this order:

        - a `TrainingManager`;
        - a `LuxonisModelSummary`;
        - a ``ModelCheckpoint`` on ``val/loss``, saved in
          ``save_dir / "min_val_loss"``;
        - an `AIMETCallback`, when ``cfg.exporter.aimet.active`` is on;
        - a ``ModelCheckpoint`` on the main metric, saved in
          ``save_dir / "best_val_metric"``, when the config has one;
        - every active callback of ``cfg.trainer.callbacks``;
        - a ``GradientAccumulationScheduler``, when
          ``cfg.trainer.accumulate_grad_batches`` is set and no
          callback of that class is present yet.

        Returns:
            ``list[pl.Callback]``: The callbacks, in that order.

        """
        return self.nodes.build_callbacks(self.save_dir)

    @override
    def configure_optimizers(
        self,
    ) -> tuple[
        Sequence[Optimizer], Sequence[LRSchedulerTypeUnion | LRSchedulerConfig]
    ]:
        """Build the optimizers and the schedulers of the run.

        Lightning calls it when a fit starts. The method builds a
        `TrainingPlan` with `resolve_training_plan`. The plan combines
        the finetuning rules of the nodes, the rules of the training
        strategy, and the trainer-level optimizer and scheduler. The
        method instantiates the plan with `build_training_plan`. It
        then hands the group handles to the training strategy and to
        the freeze schedule of the nodes. It stores the runtime in
        `training_plan`, and logs a summary of the parameter groups.

        A ``ReduceLROnPlateau`` scheduler in ``max`` mode monitors the
        main metric as ``val/metric/<task>-<node>/<metric>``. Without
        a main metric, `build_training_plan` raises ``ValueError``. In
        any other mode the scheduler monitors ``val/loss``.

        Returns:
            ``tuple[Sequence[Optimizer], Sequence[LRSchedulerTypeUnion | LRSchedulerConfig]]``:
            A list with the one optimizer of the plan, which is a
            `CompositeOptimizer` when the plan has several inner
            optimizers, and the scheduler configs of the plan.

        """
        plan = resolve_training_plan(
            self.cfg, self.nodes, self.training_strategy
        )
        runtime = build_training_plan(
            plan,
            self.cfg,
            self._main_metric_monitor(),
            self.training_strategy,
        )
        if self.training_strategy is not None:
            self.training_strategy.attach(runtime, plan.handles_by_tag)
        self.nodes.freeze_schedule.attach_group_handles(runtime)
        self._training_plan = runtime

        self._log_optimizer_scheduler_info(
            list(runtime.inner_optimizers), list(runtime.members)
        )

        return [runtime.optimizer], runtime.scheduler_configs

    @property
    def training_plan(self) -> TrainingPlanRuntime | None:
        """The runtime that `configure_optimizers` built, or ``None``.

        It holds the optimizers and the schedulers of the training plan.
        The value is ``None`` before the first `configure_optimizers`
        call.

        """
        return self._training_plan

    def _main_metric_monitor(self) -> str | None:
        if self.nodes.main_metric is None:
            return None
        node_name, metric_name = self.nodes.main_metric
        formatted_node = self.nodes.formatted_name(node_name)
        return f"val/metric/{formatted_node}/{metric_name}"

    def load_checkpoint(self, ckpt: PathType | dict[str, Any] | None) -> None:
        """Load the weights of every node from a checkpoint.

        The method loads the checkpoint file on the device of the
        module when it gets a path. It warns when
        ``cfg.trainer.resume_training`` is on and the ``trainer.epochs``
        of the checkpoint config is above ``cfg.trainer.epochs``. It
        also warns when the predefined model of the config and the one
        of the checkpoint resolve to different classes. It warns too
        when the predefined model of the checkpoint no longer resolves.

        The method splits the state dict by node. A checkpoint of
        version 0.4 or later keys a node as ``nodes.<node>.module.``,
        an older one as ``nodes.<node>.``. A checkpoint without a
        ``version`` key counts as ``0.3.0``. Each part loads into its
        node with ``strict=True``. When that fails:

        - With ``cfg.trainer.strict_weights_loading`` on, the method
          raises the error.
        - Otherwise, the method remaps the keys through the
          ``execution_order`` of the checkpoint and of the model, and
          loads the part again. Without an execution order, or when
          the remap fails, the part loads with ``strict=False`` and a
          log message reports it.

        The method runs a forward pass on zero inputs to record the
        execution order of the model, which leaves the module in
        evaluation mode.

        Args:
            ckpt (``PathType | dict[str, Any] | None``): A path to a
                checkpoint file, or a loaded checkpoint dictionary
                with a ``state_dict`` key. ``None`` loads nothing.

        Raises:
            ValueError: When the checkpoint has no ``state_dict`` key.
            RuntimeError: When a node fails to load with
                ``cfg.trainer.strict_weights_loading`` on, or when the
                checkpoint holds no key for a node.

        """
        if ckpt is None:
            return

        if isinstance(ckpt, str | Path):
            ckpt = cast(
                dict[str, Any], torch.load(ckpt, map_location=self.device)
            )  # nosemgrep

        if "state_dict" not in ckpt:
            raise ValueError("Checkpoint does not contain state_dict.")

        previous_cfg = ckpt.get("config", None)
        if self.cfg.trainer.resume_training and isinstance(previous_cfg, dict):
            self._check_valid_epoch_counts(previous_cfg)
        from luxonis_train.config.predefined_versions import (
            warn_on_predefined_model_mismatch,
        )

        warn_on_predefined_model_mismatch(
            self.cfg.model.predefined_model, ckpt.get("predefined_model")
        )

        state_dict = ckpt["state_dict"]
        ver = Version.parse(ckpt.get("version", "0.3.0"))
        strict_weights_loading = self.cfg.trainer.strict_weights_loading

        old_order = ckpt.get("execution_order")
        new_order = get_model_execution_order(self)

        for node_name, node in self.nodes.items():
            self._load_node_checkpoint(
                node_name,
                node,
                self._node_state_dict(node_name, state_dict, ver),
                strict_weights_loading,
                old_order,
                new_order,
            )

    def detach(self) -> None:
        """Detach the module from its trainer.

        The method sets ``trainer`` to ``None``. After it, `tracker`
        returns ``None`` and `progress_bar` raises ``AttributeError``.

        """
        self.trainer = None

    def _check_valid_epoch_counts(self, ckpt_config: dict) -> None:
        previous_trainer_cfg = ckpt_config.get("trainer", {})
        previous_epochs = previous_trainer_cfg.get("epochs", None)

        if (
            previous_epochs is not None
            and previous_epochs > self.cfg.trainer.epochs
        ):
            logger.warning(
                f"Checkpoint was previously trained for {previous_epochs} epochs, "
                f"but current config requests only {self.cfg.trainer.epochs} epochs. "
                "Please set a number of epochs that is higher than the previously-trained epoch number."
            )

    def _evaluation_step(
        self,
        mode: Literal["test", "val"],
        inputs: dict[str, Tensor] | Tensor,
        labels: Labels,
    ) -> dict[str, Tensor]:
        max_log_images = self.cfg.trainer.n_log_images
        if isinstance(inputs, Tensor):
            inputs = {self.image_source: inputs}
        input_image = inputs[self.image_source]

        # Smart logging is decided based on the classification task keys that are merged for all tasks
        cls_task_keys: list[str] | None = [
            k for k in labels if "/classification" in k
        ] or None
        images = None
        if self._n_logged_images < max_log_images:
            images = get_denormalized_images(self.cfg, input_image)

        outputs = self.full_forward(
            inputs,
            labels,
            images=images,
            compute_metrics=True,
            compute_visualizations=True,
        )

        _, losses = compute_losses(self.cfg, outputs.losses, self.device)

        self._loss_accumulators[mode].update(losses)

        if outputs.visualizations:
            self._log_visualizations(
                outputs, labels, cls_task_keys, mode, max_log_images
            )

        return losses

    def _evaluation_epoch_end(self, mode: Literal["test", "val"]) -> None:
        _log_accumulated_losses(self, mode)

        table, matrices = _aggregate_and_log_metrics(self, mode)

        self._print_results(
            stage="Validation" if mode == "val" else "Test",
            loss=self._loss_accumulators[mode]["loss"],
            metrics=table,
            matrices=matrices,
        )

        _flush_buffered_visualizations(self, mode)
        _reset_epoch_logging_state(self, mode)

    @rank_zero_only
    def _print_results(
        self,
        stage: str,
        loss: float,
        metrics: dict[str, dict[str, float]],
        matrices: dict[str, dict[str, dict[str, Any]]],
    ) -> None:
        """Print the loss and the metrics of an evaluation stage.

        The method runs on rank zero only. It logs the loss, prints
        the tables through `progress_bar`, and logs the value of the
        main metric when the config defines one.

        Args:
            stage (str): The name of the stage, ``"Validation"`` or
                ``"Test"``.
            loss (float): The mean total loss of the epoch.
            metrics (dict[str, dict[str, float]]): The scalar metric
                values, keyed by node name and metric name.
            matrices (``dict[str, dict[str, dict[str, Any]]]``): The
                matrix metrics, keyed by node name and metric name, in
                the format of
                `BaseLuxonisProgressBar.format_matrix_for_printing`.

        """
        logger.info(f"{stage} loss: {loss:.4f}")

        self.progress_bar.print_results(
            stage=stage, loss=loss, metrics=metrics, matrices=matrices
        )

        if self.nodes.main_metric is not None:
            node_name, metric_name = self.nodes.main_metric

            value = metrics[node_name][metric_name]
            logger.info(
                f"{stage} main metric ({node_name}/{metric_name}): {value:.4f}"
            )

    def get_mlflow_logging_keys(self) -> dict[str, list[str]]:
        """Return the metric keys and the artifact paths of a full run.

        The result predicts what the tracker receives over a run with
        the current config. `LuxonisModel.tune` reads it to find the
        monitor key of the main metric. The method calls ``compute`` on
        every metric to learn the names of its sub-metrics, without a
        reset.

        In every key, ``<task>-<node>`` is the log name of the node, see
        `Nodes.formatted_name`.

        The ``"metrics"`` list holds:

        - ``<mode>/loss`` and ``<mode>/loss/<task>-<node>/<loss>`` for
          ``train``, ``val``, and ``test``. The sub-loss keys are not
          listed;
        - ``val/metric/<task>-<node>/<name>`` and
          ``test/metric/<task>-<node>/<name>`` for every metric value
          whose name does not contain ``confusion_matrix``. The ``val``
          key is absent when the run has no validation epoch, that is,
          when ``validation_interval`` is ``-1`` and
          ``run_validation_after_first_epoch`` is off;
        - the timing keys of `TrainingProgressCallback`, when the
          config lists it.

        The ``"artifacts"`` list holds:

        - ``<mode>/metrics/<epoch>/<task>-<node>/<name>.json`` for every
          metric value whose name contains ``confusion_matrix``, at
          epoch 0 and at every validation epoch for ``val``, and at
          epoch ``cfg.trainer.epochs`` for ``test``;
        - ``<mode>/metrics/<task>-<node>/<metric>/<epoch>/<artifact>.png``
          for every artifact name of every metric, at the same epochs;
        - ``<mode>/visualizations/<task>-<node>/<visualizer>/<epoch>/<i>.png``
          for every visualizer, at the same epochs, for ``<i>`` below
          ``cfg.trainer.n_log_images``;
        - the files of the callbacks the config lists, active or not.
          ``<name>`` is ``cfg.exporter.name``, or ``cfg.model.name``
          without it:

          - ``UploadCheckpoint``: ``best_val_metric.ckpt`` and
            ``min_val_loss.ckpt``;
          - ``ExportOnTrainEnd``: ``<name>.onnx``;
          - ``ArchiveOnTrainEnd``: ``<name>.onnx.tar.xz``;
          - ``ConvertOnTrainEnd``: ``<name>.onnx`` and
            ``<name>.onnx.tar.xz``;
          - ``AIMETCallback``: ``<name>.onnx``, ``<name>.onnx.data``,
            ``<name>.onnx.tar.xz``, and ``<name>.encodings``;

        - ``luxonis_train.log``, ``training_config.yaml``, and
          ``<model name>.yaml``, where ``<model name>`` is
          ``cfg.model.name``.

        Returns:
            dict[str, list[str]]: The ``"metrics"`` and the
            ``"artifacts"`` lists, each sorted.

        """
        val_eval_epochs, test_eval_epoch = _evaluation_epochs(self.cfg)

        metric_keys = _loss_metric_keys(self)
        node_metric_keys, artifact_keys = _metric_and_artifact_keys(
            self, val_eval_epochs, test_eval_epoch
        )
        metric_keys |= node_metric_keys

        callback_metric_keys, callback_artifact_keys = _callback_keys(self.cfg)
        metric_keys |= callback_metric_keys
        artifact_keys |= callback_artifact_keys

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

    @override
    def __getstate__(self):
        state = super().__getstate__()
        state["_core"] = None
        _INTERNAL["trainer"] = self._trainer
        _INTERNAL["core"] = self._core
        return state

    @override
    def __setstate__(self, state: Any):
        super().__setstate__(state)
        self._trainer = _INTERNAL.get("trainer")  # type: ignore
        self._core = _INTERNAL.get("core")

    def _get_node_order_mapping(
        self, node_name: str, old_order: list[str], new_order: list[str]
    ) -> dict[str, str]:
        """Map the old parameter names of a node to the new ones.

        The method keeps the entries of both orders that hold
        ``.<node_name>.``, strips their prefix, and pairs them by
        position.

        Args:
            node_name (str): The identifier of the node.
            old_order (list[str]): The execution order of the
                checkpoint.
            new_order (list[str]): The execution order of the model.

        Returns:
            dict[str, str]: Each old module name of the node, without
            the ``nodes.<node>.`` or ``nodes.<node>.module.`` prefix,
            mapped to the new one.

        Raises:
            RuntimeError: When the two orders hold a different number
                of entries for the node.

        """
        old_order = [name for name in old_order if f".{node_name}." in name]
        new_order = [name for name in new_order if f".{node_name}." in name]
        if len(old_order) != len(new_order):  # pragma: no cover
            raise RuntimeError(
                "Execution order length mismatch between checkpoint and model"
            )
        return {
            self._strip_state_prefix(old_name): self._strip_state_prefix(
                new_name
            )
            for old_name, new_name in zip(old_order, new_order, strict=True)
        }

    @staticmethod
    def _strip_state_prefix(key: str) -> str:
        idx = 3 if "module." in key else 2
        return ".".join(key.split(".")[idx:])

    def _log_optimizer_scheduler_info(
        self,
        optimizers: Sequence[Optimizer],
        schedulers: Sequence[Any],
    ) -> None:
        from luxonis_train.callbacks.luxonis_progress_bar import (
            build_optimizer_summary,
            log_optimizer_summary,
        )

        summary = build_optimizer_summary(
            optimizers,
            schedulers,
            {name: node.module for name, node in self.nodes.items()},
        )
        log_optimizer_summary(summary, use_rich=self.cfg.rich_logging)

    def _get_output_onnx_names(self, inputs: dict[str, Tensor]) -> list[str]:
        outputs = self.full_forward(inputs).outputs
        export_names = self._valid_export_output_names(outputs)
        return self._render_output_names(_output_order(outputs), export_names)

    def _add_custom_data_to_checkpoint(
        self, checkpoint: dict[str, Any]
    ) -> None:
        checkpoint["state_dict"] = filter_checkpoint_state_dict(
            checkpoint["state_dict"]
        )
        predefined_model = (
            _checkpoint_predefined_model(self.cfg)
            or self._ckpt_predefined_model
        )
        checkpoint |= {
            "version": luxonis_train.__version__,
            "execution_order": get_model_execution_order(self),
            "config": self.cfg.model_dump(),
            "dataset_metadata": self.dataset_metadata.dump(),
        }
        if predefined_model is None:
            checkpoint.pop("predefined_model", None)
        else:
            checkpoint["predefined_model"] = predefined_model

    def _collect_node_results(
        self,
        node: NodeWrapper,
        node_name: str,
        outputs: Packet[Tensor],
        labels: Labels | None,
        images: Tensor | None,
        losses: _NodeLosses,
        visualizations: dict[str, dict[str, Tensor]],
        *,
        compute_loss: bool,
        compute_metrics: bool,
        compute_visualizations: bool,
    ) -> None:
        if compute_loss and node.losses and labels is not None:
            self._collect_losses(node, node_name, outputs, labels, losses)

        if compute_metrics and node.metrics and labels is not None:
            self._update_metrics(node, outputs, labels)

        if compute_visualizations and node.visualizers and images is not None:
            self._collect_visualizations(
                node, node_name, outputs, labels, images, visualizations
            )

    def _collect_losses(
        self,
        node: NodeWrapper,
        node_name: str,
        outputs: Packet[Tensor],
        labels: Labels,
        losses: _NodeLosses,
    ) -> None:
        for loss_name, loss in node.losses.items():
            loss.to(self.device)
            if self.training:
                loss.train()
            losses[node_name][loss_name] = loss.run(outputs, labels)

    def _update_metrics(
        self, node: NodeWrapper, outputs: Packet[Tensor], labels: Labels
    ) -> None:
        for metric in node.metrics.values():
            metric.to(self.device)
            metric.run_update(outputs, labels)

    def _collect_visualizations(
        self,
        node: NodeWrapper,
        node_name: str,
        outputs: Packet[Tensor],
        labels: Labels | None,
        images: Tensor,
        visualizations: dict[str, dict[str, Tensor]],
    ) -> None:
        for viz_name, visualizer in node.visualizers.items():
            visualizer.to(self.device)
            viz = combine_visualizations(
                visualizer.run(images, images, outputs, labels),
            )
            visualizations[node_name][viz_name] = viz

    def _drop_unused_outputs(
        self, computed: dict[str, Packet[Tensor]], unprocessed: list[str]
    ) -> None:
        for computed_name in list(computed.keys()):
            if computed_name in self.outputs:
                continue
            needed = any(
                computed_name in self.nodes.graph[unprocessed_name]
                for unprocessed_name in unprocessed
            )
            if not needed:
                del computed[computed_name]

    def _node_state_dict(
        self, node_name: str, state_dict: dict[str, Tensor], ver: Version
    ) -> dict[str, Tensor]:
        prefix = (
            f"nodes.{node_name}.{'module.' if ver >= Version(0, 4) else ''}"
        )
        return {
            self._strip_state_prefix(k): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }

    def _load_node_checkpoint(
        self,
        node_name: str,
        node: NodeWrapper,
        sub_state_dict: dict[str, Tensor],
        strict_weights_loading: bool,
        old_order: list[str] | None,
        new_order: list[str],
    ) -> None:
        try:
            node.module.load_checkpoint(sub_state_dict, strict=True)
        except RuntimeError:  # pragma: no cover
            logger.error(f"Failed to load checkpoint for node '{node_name}'")
            if strict_weights_loading:
                raise
            if old_order is None:
                logger.error(
                    "Execution order not found in the checkpoint. "
                    "Unable to automatically upgrade the weights."
                )
                _load_non_strict(node, sub_state_dict)
                return
            self._load_with_order_mapping(
                node_name, node, sub_state_dict, old_order, new_order
            )

    def _load_with_order_mapping(  # pragma: no cover
        self,
        node_name: str,
        node: NodeWrapper,
        sub_state_dict: dict[str, Tensor],
        old_order: list[str],
        new_order: list[str],
    ) -> None:
        try:
            order_mapping = self._get_node_order_mapping(
                node_name, old_order, new_order
            )
        except RuntimeError as e:
            logger.error(
                f"Failed to create execution order mapping for node '{node_name}'"
            )
            logger.error(str(e))
            _load_non_strict(node, sub_state_dict)
            return

        logger.info(
            f"Using execution order to transform incompatible weights for node '{node_name}'"
        )
        new_state_dict = _remap_state_dict(sub_state_dict, order_mapping)
        try:
            node.module.load_checkpoint(new_state_dict, strict=True)
            logger.info(
                f"Successfully loaded transformed checkpoint for node '{node_name}'"
            )
        except RuntimeError:
            logger.error(
                f"Failed to load transformed checkpoint for node '{node_name}'"
            )
            _load_non_strict(node, sub_state_dict)

    def _log_visualizations(
        self,
        outputs: LuxonisOutput,
        labels: Labels,
        cls_task_keys: list[str] | None,
        mode: Literal["test", "val"],
        max_log_images: int,
    ) -> None:
        if cls_task_keys is not None:
            self._log_balanced_visualizations(
                outputs, labels, cls_task_keys, mode, max_log_images
            )
        else:
            self._n_logged_images = log_sequential_images(
                self.tracker,
                self.nodes,
                outputs.visualizations,
                self._n_logged_images,
                max_log_images,
                mode,
                self.current_epoch,
            )

    def _log_balanced_visualizations(
        self,
        outputs: LuxonisOutput,
        labels: Labels,
        cls_task_keys: list[str],
        mode: Literal["test", "val"],
        max_log_images: int,
    ) -> None:
        labels_copy = _prepare_balanced_labels(labels)

        n_classes = sum(labels_copy[task].shape[1] for task in cls_task_keys)
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

    def _valid_export_output_names(
        self, outputs: dict[str, Packet[Tensor]]
    ) -> dict[str, list[str]]:
        output_counts = {
            node_name: sum(len(out) for out in outs.values())
            for node_name, outs in outputs.items()
        }
        export_names: dict[str, list[str]] = {}
        for node_name, node in self.nodes.items():
            names = node.module.export_output_names
            if names is None:
                continue
            output_count = output_counts.get(node_name, 0)
            if len(names) != output_count:
                logger.warning(
                    f"Number of provided output names for node {node_name} "
                    f"({len(names)}) does not match "
                    f"number of outputs ({output_count}). "
                    f"Using default names."
                )
                continue
            export_names[node_name] = names
        return export_names

    def _render_output_names(
        self,
        output_order: list[tuple[str, str, int]],
        export_names: dict[str, list[str]],
    ) -> list[str]:
        output_names = []
        # For cases where export_output_names should be used but
        # output node's output is split into multiple subnodes
        running_i: dict[str, int] = {}
        for node_name, output_name, i in output_order:
            names = export_names.get(node_name)
            if names is None:
                output_names.append(
                    f"{self.nodes[node_name].task_name}/{node_name}/{output_name}/{i}"
                )
                continue
            running_i[node_name] = running_i.get(node_name, -1) + 1
            output_names.append(names[running_i[node_name]])
        return output_names


def _checkpoint_predefined_model(cfg: Config) -> dict[str, Any] | None:
    """Dump a predefined model with ``latest`` resolved to a version.

    Args:
        cfg (Config): The config of the module.

    Returns:
        ``dict[str, Any] | None``: The dump of
        ``cfg.model.predefined_model``. A ``version`` of ``"latest"``
        becomes the version number of the resolved class. ``None`` when
        the config names no predefined model.

    """
    predefined_model = cfg.model.predefined_model
    if predefined_model is None:
        return None
    dumped = predefined_model.model_dump()
    if dumped.get("version") == "latest":
        from luxonis_train.config.predefined_versions import (
            resolve_predefined_class,
        )

        dumped["version"] = resolve_predefined_class(
            predefined_model.name, "latest"
        )._VERSION
    return dumped


def _node_inputs(
    input_names: list[str],
    computed: dict[str, Packet[Tensor]],
    inputs: dict[str, Tensor],
) -> list[Packet[Tensor]]:
    return [
        computed[name] if name in computed else {"features": [inputs[name]]}
        for name in input_names
    ]


def _output_order(
    outputs: dict[str, Packet[Tensor]],
) -> list[tuple[str, str, int]]:
    return sorted(
        (node_name, output_name, i)
        for node_name, outs in outputs.items()
        for output_name, out in outs.items()
        for i in range(len(out))
    )


def _prepare_balanced_labels(labels: Labels) -> Labels:
    labels_copy = {k: v.clone() for k, v in labels.items()}
    # Remove background class from segmentation tasks
    for k in (k for k in labels_copy if "/segmentation" in k):
        cls_key = f"{k[: -len('/segmentation')]}/classification"
        if cls_key not in labels_copy:
            continue
        labels_copy[cls_key] = (
            labels_copy[cls_key][:, 1:]
            if labels_copy[cls_key].shape[1] > 1
            else labels_copy[cls_key]
        )
    return labels_copy


def _format_loss_name(nodes: Nodes, name: str) -> str:
    if "/" not in name:
        return name
    node_name = name.split("/")[1]
    return name.replace(node_name, nodes.formatted_name(node_name))


def _log_accumulated_losses(module: LuxonisLightningModule, mode: str) -> None:
    for name, value in module._loss_accumulators[mode].items():
        module.log(
            f"{mode}/{_format_loss_name(module.nodes, name)}",
            value,
            sync_dist=True,
        )


def _aggregate_and_log_metrics(
    module: LuxonisLightningModule, mode: Literal["test", "val"]
) -> tuple[
    dict[str, dict[str, float]], dict[str, dict[str, dict[str, object]]]
]:
    table: defaultdict[str, dict[str, float]] = defaultdict(dict)
    matrices: defaultdict[str, dict[str, dict[str, object]]] = defaultdict(
        dict
    )

    for node_name, node in module.nodes.items():
        formatted_node_name = module.nodes.formatted_name(node_name)
        for metric_name, metric in node.metrics.items():
            computed = metric.compute()
            values = postprocess_metrics(
                metric_name,
                metric.get_loggable_values(computed),
                log_sub_metrics=module.cfg.trainer.log_sub_metrics,
            )
            if (
                module.trainer.is_global_zero
                and not module.trainer.sanity_checking
            ):
                log_metric_artifacts(
                    module.tracker,
                    metric,
                    computed,
                    mode=mode,
                    formatted_node_name=formatted_node_name,
                    metric_name=metric_name,
                    current_epoch=module.current_epoch,
                )
            metric.reset()
            _check_metrics_on_device(module, values)
            _log_metric_values(
                module,
                mode,
                node,
                node_name,
                formatted_node_name,
                values,
                table,
                matrices,
            )
    return table, matrices


def _check_metrics_on_device(
    module: LuxonisLightningModule, values: dict[str, Tensor]
) -> None:
    if isinstance(
        module.trainer.strategy, DDPStrategy
    ) and not check_tensor_device(list(values.values()), module.device):
        raise RuntimeError(
            "When using DDP all metrics must reside on the model's device"
        )


def _log_metric_values(
    module: LuxonisLightningModule,
    mode: Literal["test", "val"],
    node: NodeWrapper,
    node_name: str,
    formatted_node_name: str,
    values: dict[str, Tensor],
    table: dict[str, dict[str, float]],
    matrices: dict[str, dict[str, dict[str, object]]],
) -> None:
    for name, value in values.items():
        if value.dim() == 2:
            matrix_info = module.progress_bar.format_matrix_for_printing(
                node, name, value
            )
            module.tracker.log_matrix(
                matrix=value.cpu().numpy(),
                name=f"{mode}/metrics/{module.current_epoch}/"
                f"{formatted_node_name}/{name}",
                step=module.current_epoch,
                extra_data={"class_names": matrix_info["row_labels"]},
            )
            matrices[node_name][name] = matrix_info
        else:
            table[node_name][name] = value.cpu().item()
            module.log(
                f"{mode}/metric/{formatted_node_name}/{name}",
                value,
                sync_dist=True,
            )


def _flush_buffered_visualizations(
    module: LuxonisLightningModule, mode: Literal["test", "val"]
) -> None:
    if module._n_logged_images == module.cfg.trainer.n_log_images:
        module._needs_vis_buffering = False
        return

    logger.warning(
        f"Logged images ({module._n_logged_images}) != expected ({module.cfg.trainer.n_log_images}). Possible reasons: "
        f"class imbalance or a small number of images in the split. Trying to log more images."
    )
    for missing_visualizations in module._sequentially_logged_visualizations:
        module._n_logged_images = log_sequential_images(
            module.tracker,
            module.nodes,
            missing_visualizations,
            module._n_logged_images,
            module.cfg.trainer.n_log_images,
            mode,
            module.current_epoch,
        )


def _reset_epoch_logging_state(
    module: LuxonisLightningModule, mode: Literal["test", "val"]
) -> None:
    module._sequentially_logged_visualizations.clear()

    module._n_logged_images = 0
    if module._class_log_counts:
        module._class_log_counts = [0] * len(module._class_log_counts)
    module._loss_accumulators[mode].clear()


def _evaluation_epochs(cfg: Config) -> tuple[set[int], int]:
    val_eval_epochs = {
        max(0, i - 1)
        for i in range(
            cfg.trainer.validation_interval,
            cfg.trainer.epochs + 1,
            cfg.trainer.validation_interval,
        )
    }
    if cfg.trainer.run_validation_after_first_epoch:
        val_eval_epochs.add(0)
    return val_eval_epochs, cfg.trainer.epochs


def _loss_metric_keys(module: LuxonisLightningModule) -> set[str]:
    metric_keys: set[str] = set()
    for mode in ("train", "val", "test"):
        metric_keys.add(f"{mode}/loss")
        for node_name, node in module.nodes.items():
            formatted_node_name = module.nodes.formatted_name(node_name)
            for loss_name in node.losses:
                metric_keys.add(
                    f"{mode}/loss/{formatted_node_name}/{loss_name}"
                )
    return metric_keys


def _metric_and_artifact_keys(
    module: LuxonisLightningModule,
    val_eval_epochs: set[int],
    test_eval_epoch: int,
) -> tuple[set[str], set[str]]:
    metric_keys: set[str] = set()
    artifact_keys: set[str] = set()
    for node_name, node in module.nodes.items():
        formatted_node_name = module.nodes.formatted_name(node_name)
        for metric_name, metric in node.metrics.items():
            values = postprocess_metrics(
                metric_name,
                metric.get_loggable_values(metric.compute()),
                log_sub_metrics=module.cfg.trainer.log_sub_metrics,
            )
            for sub_name in values:
                sub_metric_keys, sub_artifact_keys = _metric_sub_name_keys(
                    sub_name,
                    formatted_node_name,
                    val_eval_epochs,
                    test_eval_epoch,
                )
                metric_keys |= sub_metric_keys
                artifact_keys |= sub_artifact_keys

            artifact_keys |= _metric_artifact_keys(
                metric.get_artifact_names(),
                formatted_node_name,
                metric_name,
                val_eval_epochs,
                test_eval_epoch,
            )

        artifact_keys |= _visualization_artifact_keys(
            node,
            formatted_node_name,
            module.cfg.trainer.n_log_images,
            val_eval_epochs,
            test_eval_epoch,
        )
    return metric_keys, artifact_keys


def _metric_sub_name_keys(
    sub_name: str,
    formatted_node_name: str,
    val_eval_epochs: set[int],
    test_eval_epoch: int,
) -> tuple[set[str], set[str]]:
    metric_keys: set[str] = set()
    artifact_keys: set[str] = set()
    if "confusion_matrix" in sub_name:
        for epoch_idx in {0, *val_eval_epochs}:
            artifact_keys.add(
                f"val/metrics/{epoch_idx}/{formatted_node_name}/{sub_name}.json"
            )
        artifact_keys.add(
            f"test/metrics/{test_eval_epoch}/{formatted_node_name}/{sub_name}.json"
        )
    else:
        if val_eval_epochs:
            metric_keys.add(f"val/metric/{formatted_node_name}/{sub_name}")
        metric_keys.add(f"test/metric/{formatted_node_name}/{sub_name}")
    return metric_keys, artifact_keys


def _metric_artifact_keys(
    artifact_names: tuple[str, ...],
    formatted_node_name: str,
    metric_name: str,
    val_eval_epochs: set[int],
    test_eval_epoch: int,
) -> set[str]:
    keys: set[str] = set()
    for artifact_name in artifact_names:
        for epoch_idx in {0, *val_eval_epochs}:
            keys.add(
                mlflow_image_key(
                    metric_artifact_image_name(
                        "val",
                        formatted_node_name,
                        metric_name,
                        artifact_name,
                    ),
                    epoch_idx,
                )
            )
        keys.add(
            mlflow_image_key(
                metric_artifact_image_name(
                    "test",
                    formatted_node_name,
                    metric_name,
                    artifact_name,
                ),
                test_eval_epoch,
            )
        )
    return keys


def _visualization_artifact_keys(
    node: NodeWrapper,
    formatted_node_name: str,
    n_log_images: int,
    val_eval_epochs: set[int],
    test_eval_epoch: int,
) -> set[str]:
    keys: set[str] = set()
    for viz_name in node.visualizers:
        for epoch_idx in {0, *val_eval_epochs}:
            for i in range(n_log_images):
                keys.add(
                    f"val/visualizations/{formatted_node_name}/{viz_name}/{epoch_idx}/{i}.png"
                )
        for i in range(n_log_images):
            keys.add(
                f"test/visualizations/{formatted_node_name}/{viz_name}/{test_eval_epoch}/{i}.png"
            )
    return keys


def _callback_keys(cfg: Config) -> tuple[set[str], set[str]]:
    model_name = cfg.exporter.name or cfg.model.name
    artifacts_by_callback: dict[str, set[str]] = {
        "UploadCheckpoint": {"best_val_metric.ckpt", "min_val_loss.ckpt"},
        "ExportOnTrainEnd": {f"{model_name}.onnx"},
        "ArchiveOnTrainEnd": {f"{model_name}.onnx.tar.xz"},
        "ConvertOnTrainEnd": {
            f"{model_name}.onnx",
            f"{model_name}.onnx.tar.xz",
        },
        "AIMETCallback": {
            f"{model_name}.onnx",
            f"{model_name}.onnx.data",
            f"{model_name}.onnx.tar.xz",
            f"{model_name}.encodings",
        },
    }
    metric_keys: set[str] = set()
    artifact_keys: set[str] = set()
    for callback in cfg.trainer.callbacks:
        artifact_keys |= artifacts_by_callback.get(callback.name, set())
        if callback.name == "TrainingProgressCallback":
            metric_keys |= _TRAINING_PROGRESS_METRIC_KEYS
    return metric_keys, artifact_keys


def _remap_state_dict(  # pragma: no cover
    sub_state_dict: dict[str, Tensor], order_mapping: dict[str, str]
) -> dict[str, Tensor]:
    new_state_dict = {}
    for old_name, value in sub_state_dict.items():
        *old_name_parts, parameter_name = old_name.split(".")

        bare_name = ".".join(old_name_parts)
        if bare_name not in order_mapping:
            logger.warning(
                f"Skipping weight {bare_name} as it is not present in the execution order of the old weights."
            )
            continue
        new_name = order_mapping[bare_name]
        new_state_dict[f"{new_name}.{parameter_name}"] = value
    return new_state_dict


def _load_non_strict(  # pragma: no cover
    node: NodeWrapper, sub_state_dict: dict[str, Tensor]
) -> None:
    logger.info(
        "Loading checkpoint with strict=False, some weights may not be loaded"
    )
    node.module.load_checkpoint(sub_state_dict, strict=False)
