from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_grad_cam import HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    SemanticSegmentationTarget,
)
from torch import Tensor

from luxonis_train.attached_modules.visualizers import (
    get_denormalized_images,
)


class ModelWrapper(pl.LightningModule):
    def __init__(self, model: pl.LightningModule, task: str) -> None:
        """Constructs `ModelWrapper`.

        @type model: pl.LightningModule
        @param model: The model to be wrapped.
        @type task: str
        @param task: The type of task (e.g., segmentation, detection,
            classification, keypoint_detection).
        """
        super().__init__()
        self.model = model
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
        input_dict = dict(image=inputs)
        output = self.model(input_dict, *args, **kwargs)

        if self.task == "segmentation":
            return output.outputs["segmentation_head"]["segmentation"][0]
        elif self.task == "detection":
            scores = output.outputs["detection_head"]["class_scores"][0]
            return scores.sum(dim=1)
        elif self.task == "classification":
            return output.outputs["classification_head"]["classification"][0]
        elif self.task == "keypoint_detection":
            scores = output.outputs["kpt_detection_head"]["class_scores"][0]
            return scores.sum(dim=1)
        else:
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
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str
    ) -> None:
        """Initializes the model wrapper.

        @type trainer: pl.Trainer
        @param trainer: The PyTorch Lightning trainer.
        @type pl_module: pl.LightningModule
        @param pl_module: The PyTorch Lightning module.
        @type stage: str
        @param stage: The stage of the training loop.
        """

        self.model = ModelWrapper(pl_module, self.task)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """At the end of first n batches, visualize the gradients using
        Grad-CAM.

        @type trainer: pl.Trainer
        @param trainer: The PyTorch Lightning trainer.
        @type pl_module: pl.LightningModule
        @param pl_module: The PyTorch Lightning module.
        @type outputs: STEP_OUTPUT
        @param outputs: The output of the model.
        @type batch: Any
        @param batch: The input batch.
        @type batch_idx: int
        @param batch_idx: The index of the batch.
        """

        if batch_idx < self.log_n_batches:
            self.visualize_gradients(trainer, pl_module, batch, batch_idx)

    def visualize_gradients(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Visualizes the gradients using Grad-CAM.

        @type trainer: pl.Trainer
        @param trainer: The PyTorch Lightning trainer.
        @type pl_module: pl.LightningModule
        @param pl_module: The PyTorch Lightning module.
        @type batch: Any
        @param batch: The input batch.
        @type batch_idx: int
        @param batch_idx: The index of the batch.
        """

        target_layers = [m[1] for m in pl_module.named_modules()][
            self.target_layer : self.target_layer + 1
        ]
        self.gradcam = HiResCAM(self.model, target_layers)

        x, y = batch
        model_input = x["image"]

        if self.task == "segmentation":
            output = self.model(model_input)
            normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
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

        images = get_denormalized_images(pl_module.cfg, x).cpu().numpy()
        for zip_idx, (image, grayscale_cam) in enumerate(
            zip(images, grayscale_cams)
        ):
            image = image / 255.0
            image = image.transpose(1, 2, 0)
            visualization = show_cam_on_image(
                image, grayscale_cam, use_rgb=True
            )
            trainer.logger.log_image(  # type: ignore
                f"gradcam/gradcam_{batch_idx}_{zip_idx}",
                visualization,
                step=trainer.global_step,
            )
