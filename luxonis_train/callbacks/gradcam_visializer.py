import pytorch_lightning as pl
import torch.nn.functional as F
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget


from luxonis_train.attached_modules.visualizers import (
    combine_visualizations,
    get_denormalized_images,
)

class SegmentationWrapper(pl.LightningModule):
    def __init__(self, model):
        super(SegmentationWrapper, self).__init__()
        self.model = model

    def forward(self, inputs, *args, **kwargs):
        # Run forward pass on the wrapped model
        inputs = dict(image=inputs)
        output = self.model(inputs, *args, **kwargs)
        # Extract and return the desired segmentation output
        return output.outputs["segmentation_head"]["segmentation"][0]



class GradCamCallback(pl.Callback):
    def __init__(self, target_layer, class_idx=None, log_every_n_steps=1):
        super().__init__()
        self.target_layer = target_layer
        self.class_idx = class_idx
        if type(self.class_idx) == int:
            self.class_idx = [self.class_idx]
        self.log_every_n_steps = log_every_n_steps
        self.gradcam = None

    def on_train_start(self, trainer, pl_module):
        # Initialize GradCam object once training starts
        print("Initializing GradCam")
        print(pl_module)
        torch.set_grad_enabled(True)
        #self.model = SegmentationWrapper(pl_module)

        #self.gradcam = GradCAM(pl_module, pl_module.nodes.ddrnet_backbone._backbone)

    
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(pl_module.nodes.ddrnet_backbone._backbone.layer4[-1].bn2)
        torch.set_grad_enabled(True)
        self.model = SegmentationWrapper(pl_module)
        #self.gradcam = GradCAM(self.model, [pl_module.nodes.ddrnet_backbone._backbone.layer4[-1].bn2])
        #self.gradcam = GradCAM(self.model, [pl_module.nodes.ddrnet_backbone.layer5[-1].bn3])
        print(list(pl_module.named_modules()))
        layer_name, layer = list(pl_module.named_modules())[30]
        self.gradcam = GradCAM(self.model, [layer])
        if self.gradcam is None:
            return
        if batch_idx % self.log_every_n_steps == 0:
            x, y = batch
            #x = x["image"]#[0]
            x = x["image"][0:1]
            #print(x["image"].shape)
            #target_category = [self.class_idx] * x["image"].size(0)
            #targets = [ClassifierOutputTarget(0)]

            output = self.model(x)
            print("output", output.shape)
            normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
            mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
            print("mask", mask)
            print("sum", np.sum(mask))


            if self.class_idx == None:
                targets = []
                for i in range(33):
                    mask_uint8 = 255 * np.uint8(mask == i)
                    mask_float = np.float32(mask == i)
                    print("sum", np.sum(mask_float))
                    print("mask_float", mask_float.shape)
                    targets.append(SemanticSegmentationTarget(i, mask_float))
            else:
                for class_idx in self.class_idx:
                    mask_uint8 = 255 * np.uint8(mask == class_idx)
                    mask_float = np.float32(mask == class_idx)
                    print("sum", np.sum(mask_float))
                    print("mask_float", mask_float.shape)
                    targets = [SemanticSegmentationTarget(class_idx, mask_float)]



            #targets = [SemanticSegmentationTarget(self.class_idx, mask_float)]
            grayscale_cam = self.gradcam(input_tensor=x, targets=targets, eigen_smooth=True)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]

            x = dict(image=x)
            images = get_denormalized_images(pl_module.cfg, x)

            image = images[0].cpu().numpy()
            image = image / 255.0
            image = image.transpose(1, 2, 0)


            #print(x["image"][0].shape)
            #image = x["image"][0].permute(1, 2, 0).cpu().numpy()
            print(np.max(image), np.min(image))
            visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)

            # Log the GradCam heatmap using trainer's logger (e.g. TensorBoard)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(visualization, cmap='jet')
            ax.axis('off')
            trainer.logger.log_image(f'gradcam/{layer_name}/gradcam_{batch_idx}', visualization, step=trainer.global_step)
            plt.close(fig)

            print(pl_module.nodes.segmentation_head.class_names)