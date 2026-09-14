"""Logs Grad-CAM heat maps of validation images.

A heat map shows the image regions that contribute to the score of one
class.

"""

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    SemanticSegmentationTarget,
)
from torch import Tensor

import luxonis_train as lxt
from luxonis_train.attached_modules.visualizers import get_denormalized_images
from luxonis_train.typing import Packet


class PLModuleWrapper(pl.LightningModule):
    """Lightning module that gives Grad-CAM one score tensor per batch.

    Grad-CAM needs a model that takes a tensor of images and returns a
    tensor of scores. `LuxonisLightningModule.full_forward` returns a
    `LuxonisOutput` with the packets of the output nodes. The wrapper
    takes the images and returns one tensor from these packets.
    `GradCamCallback` creates the wrapper.

    """

    def __init__(
        self, pl_module: "lxt.LuxonisLightningModule", task: str
    ) -> None:
        """Initialize the wrapper.

        Args:
            pl_module (LuxonisLightningModule): The model to wrap.
            task (str): Selects the output that
                `PLModuleWrapper.forward` returns. One of
                ``"segmentation"``, ``"detection"``,
                ``"classification"``, or ``"keypoints"``. The
                constructor does not check the value.

        """
        super().__init__()
        self._pl_module = pl_module
        self._task = task

    def forward(self, inputs: Tensor, *args, **kwargs) -> Tensor:
        """Run the model on images and return the scores for ``task``.

        The method calls `LuxonisLightningModule.full_forward` with
        ``{"image": inputs}`` and the extra arguments. When the model
        has more than one output node, the method logs a warning. It
        reads the packet of the first output node. The returned tensor
        depends on ``task``:

        - ``"segmentation"``: the ``"segmentation"`` output, unchanged.
        - ``"classification"``: the ``"classification"`` output,
          unchanged.
        - ``"detection"`` and ``"keypoints"``: the ``"class_scores"``
          output, summed over dimension 1. For the scores of
          `EfficientBBoxHead`, of shape ``[B, N, n_classes]`` for ``N``
          anchors, the result has the shape ``[B, n_classes]``.

        Args:
            inputs (``Tensor``): The images, of shape ``[B, C, H, W]``.
                The method passes them as the input named ``"image"``.
                The name does not follow ``loader.image_source`` of the
                config.
            *args (``Any``): Extra positional arguments for
                `LuxonisLightningModule.full_forward`.
            **kwargs (``Any``): Extra keyword arguments for
                `LuxonisLightningModule.full_forward`.

        Returns:
            ``Tensor``: The scores for ``task``.

        Raises:
            ValueError: When ``task`` is not one of the four supported
                values.

        """
        input_dict = {"image": inputs}
        output = self._pl_module.full_forward(input_dict, *args, **kwargs)
        if len(output.outputs) > 1:
            logger.warning(
                "Model has multiple heads. Using the first head for Grad-CAM."
            )
        first_head_dict = next(iter(output.outputs.values()))

        if self._task == "segmentation":
            assert isinstance(first_head_dict["segmentation"], Tensor)
            return first_head_dict["segmentation"]
        if self._task == "detection":
            scores = first_head_dict["class_scores"]
            assert isinstance(scores, Tensor)
            return scores.sum(dim=1)
        if self._task == "classification":
            assert isinstance(first_head_dict["classification"], Tensor)
            return first_head_dict["classification"]
        if self._task == "keypoints":
            scores = first_head_dict["class_scores"]
            assert isinstance(scores, Tensor)
            return scores.sum(dim=1)
        raise ValueError(f"Unknown task: {self._task}")


class GradCamCallback(pl.Callback):
    """Callback that logs Grad-CAM heat maps of validation images.

    The callback is experimental. It runs ``HiResCAM`` of
    ``pytorch_grad_cam`` on the first ``log_n_batches`` batches of each
    validation epoch. It puts each heat map over its image and logs the
    result with the tracker of the model. It logs nothing outside of
    validation.

    Grad-CAM needs gradients. The validation loop of a fit runs without
    inference mode, so the callback works there. ``trainer.validate``
    runs in inference mode by default. There, the backward pass of
    Grad-CAM raises ``RuntimeError``.

    The callback is in the ``CALLBACKS`` registry, so a config can add
    it:

    .. code-block:: yaml

        trainer:
          callbacks:
            - name: GradCamCallback
              params:
                target_layer: 10
                task: segmentation

    """

    def __init__(
        self,
        target_layer: int,
        class_idx: int = 0,
        log_n_batches: int = 1,
        task: str = "classification",
    ) -> None:
        """Initialize the callback.

        Args:
            target_layer (int): The index of the layer that Grad-CAM
                reads, in the order of ``named_modules()`` of the
                `PLModuleWrapper`. Index ``0`` is the wrapper and index
                ``1`` is the wrapped model. The callback selects the
                layer with the slice
                ``[target_layer : target_layer + 1]``. When the slice
                is empty, for example for ``-1`` or an index out of
                range, Grad-CAM raises ``ValueError``.
            class_idx (int): The index of the class that the heat maps
                explain.
            log_n_batches (int): The number of batches to log in each
                validation epoch, from the first batch.
            task (str): The type of the output to explain. One of
                ``"segmentation"``, ``"detection"``,
                ``"classification"``, or ``"keypoints"``. See
                `PLModuleWrapper.forward`.

        """
        super().__init__()
        self._target_layer = target_layer
        self._class_idx = class_idx
        self._log_n_batches = log_n_batches
        self._task = task

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str,
    ) -> None:
        """Wrap the model for Grad-CAM.

        Lightning calls this hook at the start of every stage. The hook
        creates a new `PLModuleWrapper` of ``pl_module`` and ``task`` for
        `visualize_gradients`.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (LuxonisLightningModule): The model to wrap.
            stage (str): The stage that starts. Unused.

        """
        self._pl_module = PLModuleWrapper(pl_module, self._task)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        outputs: STEP_OUTPUT,
        batch: tuple[dict[str, Tensor], Packet[Tensor]],
        batch_idx: int,
    ) -> None:
        """Log the heat maps of the first ``log_n_batches`` batches.

        Lightning calls this hook after every validation batch. When
        ``batch_idx`` is lower than ``log_n_batches``, the hook takes
        the inputs of the batch. From a dictionary of inputs, it takes
        the entry ``pl_module.image_source``. Then it calls
        `GradCamCallback.visualize_gradients`. For a later batch, the
        hook does nothing.

        Args:
            trainer (``pl.Trainer``): The trainer. It gives the step of
                the logged images.
            pl_module (LuxonisLightningModule): The model. It gives
                ``image_source``, the config, and the tracker.
            outputs (``STEP_OUTPUT``): The output of the validation
                step. Unused.
            batch (``tuple[dict[str, Tensor], Packet[Tensor]]``): The
                inputs and the labels of the batch.
            batch_idx (int): The index of the batch in the validation
                epoch.

        """
        if batch_idx < self._log_n_batches:
            images = batch[0]
            if isinstance(images, dict):
                images = images[pl_module.image_source]
            self.visualize_gradients(trainer, pl_module, images, batch_idx)

    def visualize_gradients(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        images: Tensor,
        batch_idx: int,
    ) -> None:
        """Compute the Grad-CAM heat maps of a batch and log them.

        The method creates a ``HiResCAM`` on the layer at index
        ``target_layer`` in ``named_modules()`` of the
        `PLModuleWrapper`. ``HiResCAM`` sets the wrapper and the model
        to eval mode. The method keeps the ``HiResCAM``, so its hooks
        stay on the layer until a later call replaces it. Until then,
        the hooks keep a CPU copy of the output of the layer from each
        forward pass, and of its gradient from each backward pass.

        The Grad-CAM target of each image depends on ``task``:

        - ``"segmentation"``: the method first runs the model on the
          images and applies a softmax over the classes. The mask of an
          image holds the pixels where ``class_idx`` has the highest
          probability. The target is the sum of the ``class_idx`` score
          map over that mask.
        - Any other task: the target is the ``class_idx`` entry of the
          scores from `PLModuleWrapper.forward`.

        The method computes the heat maps with gradients enabled.
        Grad-CAM clears the gradients of the model and runs a backward
        pass, so the parameters of the model keep new ``grad`` values.
        The method denormalizes the images with
        ``trainer.preprocessing.normalize`` of ``pl_module.cfg``. Each
        heat map goes over its image as a JET color map at half
        opacity. The tracker of ``pl_module`` logs each result as
        ``gradcam/gradcam_<batch_idx>_<i>`` at ``trainer.global_step``.
        ``<i>`` is the index of the image in the batch.

        Args:
            trainer (``pl.Trainer``): The trainer. It gives the step of
                the logged images.
            pl_module (LuxonisLightningModule): The model. It gives the
                config for the denormalization and the tracker.
            images (``Tensor``): The normalized images, of shape
                ``[B, C, H, W]``.
            batch_idx (int): The index of the batch. It is part of the
                image names.

        """
        target_layers = [m[1] for m in self._pl_module.named_modules()][
            self._target_layer : self._target_layer + 1
        ]
        self._gradcam = HiResCAM(self._pl_module, target_layers)

        model_input = images.clone()

        if self._task == "segmentation":
            output = self._pl_module(model_input)
            normalized_masks = F.softmax(output, dim=1).cpu()
            mask = normalized_masks.argmax(dim=1).detach().cpu().numpy()
            mask_float = (mask == self._class_idx).astype(np.float32)
            targets = [
                SemanticSegmentationTarget(self._class_idx, mask_float[i])
                for i in range(mask_float.shape[0])
            ]
        else:
            targets = [
                ClassifierOutputTarget(self._class_idx)
            ] * model_input.size(0)

        with torch.enable_grad():
            grayscale_cams = self._gradcam(
                input_tensor=model_input,
                targets=targets,  # type: ignore
            )

        np_images = (
            get_denormalized_images(pl_module.cfg, images).cpu().numpy()
        )
        for zip_idx, (image, grayscale_cam) in enumerate(
            zip(np_images, grayscale_cams, strict=True)
        ):
            image = image / 255.0
            image = image.transpose(1, 2, 0)
            visualization = show_cam_on_image(
                image, grayscale_cam, use_rgb=True
            )
            pl_module.tracker.log_image(
                f"gradcam/gradcam_{batch_idx}_{zip_idx}",
                visualization,
                step=trainer.global_step,
            )
