from typing import Literal

import torchvision
from luxonis_ml.typing import Kwargs
from torch import Tensor
from torchvision.models import ResNet as TorchResNet
from typing_extensions import override

from luxonis_train.nodes.base_node import BaseNode


class ResNet(BaseNode):
    """ResNet backbone."""

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

        Implements the backbone of a ResNet (Residual Network)
        architecture.

        ResNet is designed to address the vanishing gradient problem in
        deep neural networks by introducing skip connections. These
        connections allow the network to learn residual functions with
        reference to the layer inputs, enabling training of much deeper
        networks.

        This backbone can be used as a feature extractor for various
        computer vision tasks such as image classification, object
        detection, and semantic segmentation. It provides a robust set
        of features that can be fine-tuned for specific applications.

        The architecture consists of stacked residual blocks, each
        containing convolutional layers, batch normalization, and ReLU
        activations. The skip connections can be either identity
        mappings or projections, depending on the block type.

        Source: `https://pytorch.org/vision/main/models/resnet.html
        <https://pytorch.org/vision/main/models/resnet.html>`_

        Args:
            variant: ResNet depth. Defaults to "18".
            zero_init_residual: Zero-initialize the last BN in each
                residual branch, so that the residual branch starts with
                zeros, and each residual block behaves like an identity.
                This improves the model by 0.2~0.3% according to
                `Accurate, Large Minibatch SGD: Training ImageNet in 1
                Hour <https://arxiv.org/abs/1706.02677>`_. Defaults to
                ``False``.
            groups: Number of groups for each block. Defaults to 1. Can
                be set to a different value only for ResNet-50,
                ResNet-101, and ResNet-152. The width of the
                convolutional blocks is computed as
                ``int(in_channels * (width_per_group / 64.0)) * groups``
            width_per_group: Number of channels per group. Defaults to
                64. Can be set to a different value only for ResNet-50,
                ResNet-101, and ResNet-152. The width of the
                convolutional blocks is computed as
                ``int(in_channels * (width_per_group / 64.0)) * groups``
            replace_stride_with_dilation: Tuple of booleans where each
                indicates if the 2x2 strides should be replaced with a
                dilated convolution instead. Defaults to
                (False, False, False). Can be set to a different value
                only for ResNet-50, ResNet-101, and ResNet-152.
            weights: Whether to load pretrained weights.
            **kwargs: Base node arguments.

        License:
            `PyTorch <https://github.com/pytorch/pytorch/blob/master/LICENSE>`_

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
        r"""Extract a multi-scale ResNet feature pyramid.

        Args:
            inputs: Image batch to encode.

        Returns:
            Feature maps from the configured output stages, ordered by
            increasing depth.

        .. shape-contract::

            Inputs
                :math:`\mathrm{inputs}`
                    :math:`(B, C_{\mathrm{in}}, H_{\mathrm{in}}, W_{\mathrm{in}})`

            Outputs
                :math:`\mathrm{features}_{i}` (:math:`i = 0, \ldots, N - 1`)
                    :math:`(B, C_{i}, H_{i}, W_{i})`

            Symbols
                :math:`N`
                    Number of tensors in the feature sequence.

        """  # noqa: E501
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
