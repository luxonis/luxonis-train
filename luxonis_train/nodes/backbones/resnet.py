from typing import Literal

import torchvision
from luxonis_ml.typing import Kwargs
from torch import Tensor
from torchvision.models import ResNet as TorchResNet
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode


class ResNet(BaseNode):
    def __init__(
        self,
        variant: Literal["18", "34", "50", "101", "152"] = "18",
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: tuple[bool, bool, bool] = (
            False,
            False,
            False,
        ),
        weights: Literal["download", "none"] | None = None,
        **kwargs,
    ):
        """ResNet backbone.

        Implements the backbone of a ResNet (Residual Network) architecture.

        ResNet is designed to address the vanishing gradient problem in deep neural networks
        by introducing skip connections. These connections allow the network to learn
        residual functions with reference to the layer inputs, enabling training of much
        deeper networks.

        This backbone can be used as a feature extractor for various computer vision tasks
        such as image classification, object detection, and semantic segmentation. It
        provides a robust set of features that can be fine-tuned for specific applications.

        The architecture consists of stacked residual blocks, each containing convolutional
        layers, batch normalization, and ReLU activations. The skip connections can be
        either identity mappings or projections, depending on the block type.

        Source: U{https://pytorch.org/vision/main/models/resnet.html}

        @license: U{PyTorch<https://github.com/pytorch/pytorch/blob/master/LICENSE>}

        @param variant: ResNet variant, determining the depth and structure of the network. Options are:
            - "18": 18 layers, uses basic blocks, smaller model suitable for simpler tasks.
            - "34": 34 layers, uses basic blocks, good balance of depth and computation.
            - "50": 50 layers, introduces bottleneck blocks, deeper feature extraction.
            - "101": 101 layers, uses bottleneck blocks, high capacity for complex tasks.
            - "152": 152 layers, deepest variant, highest capacity but most computationally intensive.
            The number in each variant represents the total number of weighted layers.
            Deeper networks generally offer higher accuracy but require more computation.
        @type variant: Literal["18", "34", "50", "101", "152"]
        @default variant: "18"

        @type zero_init_residual: bool
        @param zero_init_residual: Zero-initialize the last BN in each residual branch,
            so that the residual branch starts with zeros, and each residual block behaves like an identity.
            This improves the model by 0.2~0.3% according to U{Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour <https://arxiv.org/abs/1706.02677>}. Defaults to C{False}.

        @type groups: int
        @param groups: Number of groups for each block.
            Defaults to 1. Can be set to a different value only
            for ResNet-50, ResNet-101, and ResNet-152.
            The width of the convolutional blocks is computed as
            C{int(in_channels * (width_per_group / 64.0)) * groups}

        @type width_per_group: int
        @param width_per_group: Number of channels per group.
            Defaults to 64. Can be set to a different value only
            for ResNet-50, ResNet-101, and ResNet-152.
            The width of the convolutional blocks is computed as
            C{int(in_channels * (width_per_group / 64.0)) * groups}

        @type replace_stride_with_dilation: tuple[bool, bool, bool]
        @param replace_stride_with_dilation: Tuple of booleans where each
            indicates if the 2x2 strides should be replaced with a dilated convolution instead.
            Defaults to (False, False, False). Can be set to a different value only for ResNet-50, ResNet-101, and ResNet-152.
        """
        super().__init__(**kwargs)
        self.backbone = self._get_backbone(
            variant,
            weights="DEFAULT" if weights == "download" else None,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

    def forward(self, inputs: Tensor) -> list[Tensor]:
        outs: list[Tensor] = []
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        outs.append(x)
        x = self.backbone.layer2(x)
        outs.append(x)
        x = self.backbone.layer3(x)
        outs.append(x)
        x = self.backbone.layer4(x)
        outs.append(x)

        return outs

    @staticmethod
    @override
    def get_variants() -> tuple[str, dict[str, Kwargs]]:
        return "18", {
            "18": {"variant": "18"},
            "34": {"variant": "34"},
            "50": {"variant": "50"},
            "101": {"variant": "101"},
            "152": {"variant": "152"},
        }

    @staticmethod
    def _get_backbone(
        variant: Literal["18", "34", "50", "101", "152"], **kwargs
    ) -> TorchResNet:
        variants = {
            "18": torchvision.models.resnet18,
            "34": torchvision.models.resnet34,
            "50": torchvision.models.resnet50,
            "101": torchvision.models.resnet101,
            "152": torchvision.models.resnet152,
        }
        if variant not in variants:
            raise ValueError(
                "ResNet model variant should be in "
                f"{list(variants.keys())}, got {variant}."
            )
        return variants[variant](**kwargs)
