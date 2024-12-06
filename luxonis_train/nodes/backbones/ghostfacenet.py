# Original source: https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from luxonis_train.nodes.base_node import BaseNode


def _make_divisible(v, divisor, min_value=None):
    """This function is taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return F.relu6(x + 3.0) / 6.0


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_chs,
        se_ratio=0.25,
        reduced_base_chs=None,
        act_layer=nn.PReLU,
        gate_fn=hard_sigmoid,
        divisor=4,
        **_,
    ):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible(
            (reduced_base_chs or in_chs) * se_ratio, divisor
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(
        self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.PReLU
    ):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(
            in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class ModifiedGDC(nn.Module):
    def __init__(
        self, image_size, in_chs, num_classes, dropout, emb=512
    ):  # dropout implementation is in the original code but not in the paper
        super(ModifiedGDC, self).__init__()

        if image_size % 32 == 0:
            self.conv_dw = nn.Conv2d(
                in_chs,
                in_chs,
                kernel_size=(image_size // 32),
                groups=in_chs,
                bias=False,
            )
        else:
            self.conv_dw = nn.Conv2d(
                in_chs,
                in_chs,
                kernel_size=(image_size // 32 + 1),
                groups=in_chs,
                bias=False,
            )
        self.bn1 = nn.BatchNorm2d(in_chs)
        self.dropout = nn.Dropout(dropout)

        self.conv = nn.Conv2d(in_chs, emb, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(emb)
        self.linear = (
            nn.Linear(emb, num_classes) if num_classes else nn.Identity()
        )

    def forward(self, inps):
        x = inps
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.dropout(x)
        # # Add spots to the features
        # x = torch.cat([x, spots.view(spots.size(0), -1, 1, 1)], dim=1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.bn2(x)
        x = self.linear(x)
        return x


class GhostModuleV2(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        prelu=True,
        mode=None,
        args=None,
    ):
        super(GhostModuleV2, self).__init__()
        self.mode = mode
        self.gate_fn = nn.Sigmoid()

        if self.mode in ["original"]:
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(init_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(new_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
        elif self.mode in ["attn"]:  # DFC
            self.oup = oup
            init_channels = math.ceil(oup / ratio)
            new_channels = init_channels * (ratio - 1)
            self.primary_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    init_channels,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(init_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.cheap_operation = nn.Sequential(
                nn.Conv2d(
                    init_channels,
                    new_channels,
                    dw_size,
                    1,
                    dw_size // 2,
                    groups=init_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(new_channels),
                nn.PReLU() if prelu else nn.Sequential(),
            )
            self.short_conv = nn.Sequential(
                nn.Conv2d(
                    inp, oup, kernel_size, stride, kernel_size // 2, bias=False
                ),
                nn.BatchNorm2d(oup),
                nn.Conv2d(
                    oup,
                    oup,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=oup,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
                nn.Conv2d(
                    oup,
                    oup,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=oup,
                    bias=False,
                ),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.mode in ["original"]:
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :]
        elif self.mode in ["attn"]:
            res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
            x1 = self.primary_conv(x)
            x2 = self.cheap_operation(x1)
            out = torch.cat([x1, x2], dim=1)
            return out[:, : self.oup, :, :] * F.interpolate(
                self.gate_fn(res),
                size=(out.shape[-2], out.shape[-1]),
                mode="nearest",
            )


class GhostBottleneckV2(nn.Module):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        act_layer=nn.PReLU,
        se_ratio=0.0,
        layer_id=None,
        args=None,
    ):
        super(GhostBottleneckV2, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        if layer_id <= 1:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, prelu=True, mode="original", args=args
            )
        else:
            self.ghost1 = GhostModuleV2(
                in_chs, mid_chs, prelu=True, mode="attn", args=args
            )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        self.ghost2 = GhostModuleV2(
            mid_chs, out_chs, prelu=False, mode="original", args=args
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
        if self.se is not None:
            x = self.se(x)
        x = self.ghost2(x)
        x += self.shortcut(residual)
        return x


# NODES.register_module()
class GhostFaceNetsV2(BaseNode[torch.Tensor, list[torch.Tensor]]):
    def unwrap(self, inputs):
        return [inputs[0]["features"][0]]

    def wrap(self, outputs):
        return {"features": [outputs]}

    def set_export_mode(self, mode: bool = True):
        self.export_mode = mode
        self.train(not mode)

    def __init__(
        self,
        cfgs=None,
        embedding_size=512,
        num_classes=0,
        width=1.0,
        dropout=0.2,
        block=GhostBottleneckV2,
        add_pointwise_conv=False,
        bn_momentum=0.9,
        bn_epsilon=1e-5,
        init_kaiming=True,
        block_args=None,
        *args,
        **kwargs,
    ):
        """GhostFaceNetsV2 backbone.

        GhostFaceNetsV2 is a convolutional neural network architecture focused on face recognition, but it is
        adaptable to generic embedding tasks. It is based on the GhostNet architecture and uses Ghost BottleneckV2 blocks.

        Source: U{https://github.com/Hazqeel09/ellzaf_ml/blob/main/ellzaf_ml/models/ghostfacenetsv2.py}

        @license: U{MIT License
            <https://github.com/Hazqeel09/ellzaf_ml/blob/main/LICENSE>}

        @see: U{GhostFaceNets: Lightweight Face Recognition Model From Cheap Operations
            <https://www.researchgate.net/publication/369930264_GhostFaceNets_Lightweight_Face_Recognition_Model_from_Cheap_Operations>}

        @type cfgs: list[list[list[int]]] | None
        @param cfgs: List of Ghost BottleneckV2 configurations. Defaults to None, which uses the original GhostFaceNetsV2 configuration.
        @type embedding_size: int
        @param embedding_size: Size of the embedding. Defaults to 512.
        @type num_classes: int
        @param num_classes: Number of classes. Defaults to 0, which makes the network output the raw embeddings. Otherwise it can be used to
            add another linear layer to the network, which is useful for training using ArcFace or similar classification-based losses that
            require the user to drop the last layer of the network.
        @type width: float
        @param width: Width multiplier. Increases complexity and number of parameters. Defaults to 1.0.
        @type dropout: float
        @param dropout: Dropout rate. Defaults to 0.2.
        @type block: nn.Module
        @param block: Ghost BottleneckV2 block. Defaults to GhostBottleneckV2.
        @type add_pointwise_conv: bool
        @param add_pointwise_conv: If True, adds a pointwise convolution layer at the end of the network. Defaults to False.
        @type bn_momentum: float
        @param bn_momentum: Batch normalization momentum. Defaults to 0.9.
        @type bn_epsilon: float
        @param bn_epsilon: Batch normalization epsilon. Defaults to 1e-5.
        @type init_kaiming: bool
        @param init_kaiming: If True, initializes the weights using the Kaiming initialization. Defaults to True.
        @type block_args: dict
        @param block_args: Arguments to pass to the block. Defaults to None.
        """
        # kwargs['_tasks'] = {TaskType.LABEL: 'features'}
        super().__init__(*args, **kwargs)

        inp_shape = kwargs["input_shapes"][0]["features"][0]
        # spots_shape = kwargs['input_shapes'][0]['features'][1]

        image_size = inp_shape[2]
        channels = inp_shape[1]
        if cfgs is None:
            self.cfgs = [
                # k, t, c, SE, s
                [[3, 16, 16, 0, 1]],
                [[3, 48, 24, 0, 2]],
                [[3, 72, 24, 0, 1]],
                [[5, 72, 40, 0.25, 2]],
                [[5, 120, 40, 0.25, 1]],
                [[3, 240, 80, 0, 2]],
                [
                    [3, 200, 80, 0, 1],
                    [3, 184, 80, 0, 1],
                    [3, 184, 80, 0, 1],
                    [3, 480, 112, 0.25, 1],
                    [3, 672, 112, 0.25, 1],
                ],
                [[5, 672, 160, 0.25, 2]],
                [
                    [5, 960, 160, 0, 1],
                    [5, 960, 160, 0.25, 1],
                    [5, 960, 160, 0, 1],
                    [5, 960, 160, 0.25, 1],
                ],
            ]
        else:
            self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(
            channels, output_channel, 3, 2, 1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.PReLU()
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        layer_id = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                if block == GhostBottleneckV2:
                    layers.append(
                        block(
                            input_channel,
                            hidden_channel,
                            output_channel,
                            k,
                            s,
                            se_ratio=se_ratio,
                            layer_id=layer_id,
                            args=block_args,
                        )
                    )
                input_channel = output_channel
                layer_id += 1
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(
            nn.Sequential(ConvBnAct(input_channel, output_channel, 1))
        )

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        pointwise_conv = []
        if add_pointwise_conv:
            pointwise_conv.append(
                nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
            )
            pointwise_conv.append(nn.BatchNorm2d(output_channel))
            pointwise_conv.append(nn.PReLU())
        else:
            pointwise_conv.append(nn.Sequential())

        self.pointwise_conv = nn.Sequential(*pointwise_conv)
        self.classifier = ModifiedGDC(
            image_size, output_channel, num_classes, dropout, embedding_size
        )

        # Initialize weights
        for m in self.modules():
            if init_kaiming:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    negative_slope = 0.25  # Default value for PReLU in PyTorch, change it if you use custom value
                    m.weight.data.normal_(
                        0, math.sqrt(2.0 / (fan_in * (1 + negative_slope**2)))
                    )
            if isinstance(m, nn.BatchNorm2d):
                m.momentum, m.eps = bn_momentum, bn_epsilon

    def forward(self, inps):
        x = inps[0]
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.pointwise_conv(x)
        x = self.classifier(x)
        return x

    # @property
    # def task(self) -> str:
    #     return "label"

    # @property
    # def tasks(self) -> dict:
    #     return [TaskType.LABEL]


if __name__ == "__main__":
    W, H = 256, 256
    model = GhostFaceNetsV2(image_size=W)
    model.eval()  # Set the model to evaluation mode

    # Create a dummy input tensor of the appropriate size
    x = torch.randn(1, 3, H, W)

    # Export the model
    onnx_path = "ghostfacenet.onnx"
    torch.onnx.export(
        model,  # model being run
        x,  # model input (or a tuple for multiple inputs)
        onnx_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        opset_version=12,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
        #                 'output' : {0 : 'batch_size'}}
    )
    import os

    import numpy as np
    import onnx
    import onnxsim

    # logger.info("Simplifying ONNX model...")
    model_onnx = onnx.load(onnx_path)
    onnx_model, check = onnxsim.simplify(model_onnx)
    if not check:
        raise RuntimeError("Onnx simplify failed.")
    onnx.save(onnx_model, onnx_path)

    # Add calibration data
    dir = "shared_with_container/calibration_data/"
    for file in os.listdir(dir):
        os.remove(dir + file)
    for i in range(20):
        np_array = np.random.rand(1, 3, H, W).astype(np.float32)
        np.save(f"{dir}{i:02d}.npy", np_array)
        np_array.tofile(f"{dir}{i:02d}.raw")

    # Test backpropagation on the model
    # Create a dummy target tensor of the appropriate size
    Y = model(x)
    target = torch.randn(1, 512)
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(Y, target)
    model.zero_grad()
    loss.backward()
    print("Backpropagation test successful")
