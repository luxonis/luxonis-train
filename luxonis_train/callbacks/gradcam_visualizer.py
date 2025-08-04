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
    def __init__(
        self, pl_module: "lxt.LuxonisLightningModule", task: str
    ) -> None:
        """Constructs `ModelWrapper`.

        @type pl_module: LuxonisLightningModule
        @param pl_module: The model to be wrapped.
        @type task: str
        @param task: The type of task (e.g., segmentation, detection,
            classification, keypoint_detection).
        """
        super().__init__()
        self.pl_module = pl_module
        self.task = task

    def forward(self, inputs: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass through the model, returning the output based on
        the task type.

        @type inputs: Tensor
        @param inputs: Input tensor for the model.
        @type args: Any
        @param args: Additional positional arguments.
        @type kwargs: Any
        @param kwargs: Additional keyword arguments.
        @rtype: Tensor
        @return: The processed output based on the task type.
        """
        input_dict = {"image": inputs}
        output = self.pl_module(input_dict, *args, **kwargs)
        if len(output.outputs) > 1:
            logger.warning(
                "Model has multiple heads. Using the first head for Grad-CAM."
            )
        first_head_dict = next(iter(output.outputs.values()))

        if self.task == "segmentation":
            assert isinstance(first_head_dict["segmentation"], Tensor)
            return first_head_dict["segmentation"]
        if self.task == "detection":
            scores = first_head_dict["class_scores"]
            assert isinstance(scores, Tensor)
            return scores.sum(dim=1)
        if self.task == "classification":
            assert isinstance(first_head_dict["classification"], Tensor)
            return first_head_dict["classification"]
        if self.task == "keypoints":
            scores = first_head_dict["class_scores"]
            assert isinstance(scores, Tensor)
            return scores.sum(dim=1)
        raise ValueError(f"Unknown task: {self.task}")


class GradCamCallback(pl.Callback):
    """Callback to visualize gradients using Grad-CAM (experimental).

    Works only during validation.
    """

    def __init__(
        self,
        target_layer: int,
        class_idx: int = 0,
        log_n_batches: int = 1,
        task: str = "classification",
    ) -> None:
        """Constructs `GradCamCallback`.

        @type target_layer: int
        @param target_layer: Layer to visualize gradients.
        @type class_idx: int | None
        @param class_idx: Index of the class for visualization. Defaults
            to None.
        @type log_n_batches: int
        @param log_n_batches: Number of batches to log. Defaults to 1.
        @type task: str
        @param task: The type of task. Defaults to "classification".
        """
        super().__init__()
        self.target_layer = target_layer
        self.class_idx = class_idx
        self.log_n_batches = log_n_batches
        self.task = task

    def setup(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        stage: str,
    ) -> None:
        """Initializes the model wrapper.

        @type trainer: pl.Trainer
        @param trainer: The PyTorch Lightning trainer.
        @type pl_module: LuxonisLightningModule
        @param pl_module: The LuxonisLightningModule.
        @type stage: str
        @param stage: The stage of the training loop.
        """
        self.pl_module = PLModuleWrapper(pl_module, self.task)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        outputs: STEP_OUTPUT,
        batch: tuple[dict[str, Tensor], Packet[Tensor]],
        batch_idx: int,
    ) -> None:
        """At the end of first n batches, visualize the gradients using
        Grad-CAM.

        @type trainer: pl.Trainer
        @param trainer: The PyTorch Lightning trainer.
        @type pl_module: LuxonisLightningModule
        @param pl_module: The PyTorch Lightning module.
        @type outputs: STEP_OUTPUT
        @param outputs: The output of the model.
        @type batch: Any
        @param batch: The input batch.
        @type batch_idx: int
        @param batch_idx: The index of the batch.
        """
        if batch_idx < self.log_n_batches:
            images = batch[0][pl_module.image_source]
            self.visualize_gradients(trainer, pl_module, images, batch_idx)

    def visualize_gradients(
        self,
        trainer: pl.Trainer,
        pl_module: "lxt.LuxonisLightningModule",
        images: Tensor,
        batch_idx: int,
    ) -> None:
        """Visualizes the gradients using Grad-CAM.

        @type trainer: pl.Trainer
        @param trainer: The PyTorch Lightning trainer.
        @type pl_module: pl.LightningModule
        @param pl_module: The PyTorch Lightning module.
        @type images: Tensor
        @param images: The input images.
        @type batch_idx: int
        @param batch_idx: The index of the batch.
        """
        target_layers = [m[1] for m in self.pl_module.named_modules()][
            self.target_layer : self.target_layer + 1
        ]
        self.gradcam = HiResCAM(self.pl_module, target_layers)

        model_input = images.clone()

        if self.task == "segmentation":
            output = self.pl_module(model_input)
            normalized_masks = F.softmax(output, dim=1).cpu()
            mask = normalized_masks.argmax(dim=1).detach().cpu().numpy()
            mask_float = (mask == self.class_idx).astype(np.float32)
            targets = [
                SemanticSegmentationTarget(self.class_idx, mask_float[i])
                for i in range(mask_float.shape[0])
            ]
        else:
            targets = [
                ClassifierOutputTarget(self.class_idx)
            ] * model_input.size(0)

        with torch.enable_grad():
            grayscale_cams = self.gradcam(
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
