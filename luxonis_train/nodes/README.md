# Nodes

Nodes are the basic building structures of the model. They can be connected together
arbitrarily as long as the two nodes are compatible with each other. We've grouped together nodes that are similar, so it's easier to build an architecture that makes sense.

## Table Of Contents

- [Backbones](#backbones)
  - [`ResNet`](#resnet)
  - [`MicroNet`](#micronet)
  - [`RepVGG`](#repvgg)
  - [`EfficientRep`](#efficientrep)
  - [`RexNetV1_lite`](#rexnetv1_lite)
  - [`MobileOne`](#mobileone)
  - [`MobileNetV2`](#mobilenetv2)
  - [`EfficientNet`](#efficientnet)
  - [`ContextSpatial`](#contextspatial)
  - [`DDRNet`](#ddrnet)
- [Necks](#necks)
  - [`RepPANNeck`](#reppanneck)
- [Heads](#heads)
  - [`ClassificationHead`](#classificationhead)
  - [`SegmentationHead`](#segmentationhead)
  - [`BiSeNetHead`](#bisenethead)
  - [`EfficientBBoxHead`](#efficientbboxhead)
  - [`EfficientKeypointBBoxHead`](#efficientkeypointbboxhead)
  - [`DDRNetSegmentationHead`](#ddrnetsegmentationhead)

Every node takes these parameters:

| Key                | Type          | Default value | Description                                                                 |
| ------------------ | ------------- | ------------- | --------------------------------------------------------------------------- |
| `n_classes`        | `int \| None` | `None`        | Number of classes in the dataset. Inferred from the dataset if not provided |
| `remove_on_export` | `bool`        | `False`       | Whether node should be removed when exporting the whole model               |

In addition, the following class attributes can be overridden:

| Key            | Type                                                              | Default value | Description                                                                                                                                                                                                                     |
| -------------- | ----------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attach_index` | `int \| "all" \| tuple[int, int] \| tuple[int, int, int] \| None` | `None`        | Index of previous output that the head attaches to. Each node has a sensible default. Usually should not be manually set in most cases. Can be either a single index, a slice (negative indexing is also supported), or `"all"` |
| `tasks`        | `list[TaskType] \| Dict[TaskType, str] \| None`                   | `None`        | Tasks supported by the node. Should be overridden for head nodes. Either a list of tasks or a dictionary mapping tasks to their default names                                                                                   |

Additional parameters for specific nodes are listed below.

## Backbones

### `ResNet`

Adapted from [here](https://pytorch.org/vision/main/models/resnet.html).

**Parameters:**

| Key                            | Type                                      | Default value           | Description                                                                                                           |
| ------------------------------ | ----------------------------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `variant`                      | `Literal["18", "34", "50", "101", "152"]` | `"18"`                  | Variant of the network                                                                                                |
| `download_weights`             | `bool`                                    | `False`                 | If True download weights from ImageNet                                                                                |
| `zero_init_residual`           | `bool`                                    | `False`                 | If True, the last residual block is initialized with zeros                                                            |
| `groups`                       | `int`                                     | `1`                     | Number of groups for the 3x3 convolution                                                                              |
| `width_per_group`              | `int`                                     | `64`                    | Number of channels per group for the 3x3 convolution                                                                  |
| `replace_stride_with_dilation` | `tuple[bool, bool, bool]`                 | `(False, False, False)` | Replace stride with dilation in the last three stages. Each element in the list corresponds to the last three stages. |

### `MicroNet`

Adapted from [here](https://github.com/liyunsheng13/micronet).

**Parameters:**

| Key           | Type                        | Default value | Description                                                            |
| ------------- | --------------------------- | ------------- | ---------------------------------------------------------------------- |
| `variant`     | `Literal["M1", "M2", "M3"]` | `"M1"`        | Variant of the network                                                 |
| `out_indices` | `list[int] \| None`         | `None`        | Indices of the output layers. If provided, overrides the variant value |

### `RepVGG`

Adapted from [here](https://github.com/DingXiaoH/RepVGG).

**Parameters:**

| Key                   | Type                                        | Default value | Description                                                               |
| --------------------- | ------------------------------------------- | ------------- | ------------------------------------------------------------------------- |
| `variant`             | `Literal["A0", "A1", "A2"]`                 | `"A0"`        | Variant of the network                                                    |
| `n_blocks`            | `tuple[int, int, int, int] \| None`         | `None`        | Number of blocks in each stage. If provided, overrides the variant value  |
| `width_multiplier`    | `tuple[float, float, float, float] \| None` | `None`        | Width multiplier for each stage. If provided, overrides the variant value |
| `override_groups_map` | `dict[int, int] \| None`                    | `None`        | Dictionary mapping layer to the number of groups                          |
| `use_se`              | `bool`                                      | `False`       | Whether to use `Squeeze-and-Excitation` blocks                            |
| `use_checkpoint`      | `bool`                                      | `False`       | Whether to use checkpointing for memory optimization                      |

### `EfficientRep`

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key             | Type                                                              | Default value               | Description                                                                |
| --------------- | ----------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------- |
| `variant`       | `Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]` | `"nano"`                    | Variant of the network                                                     |
| `channels_list` | `list[int]`                                                       | \[64, 128, 256, 512, 1024\] | List of number of channels for each block                                  |
| `n_repeats`     | `list[int]`                                                       | \[1, 6, 12, 18, 6\]         | List of number of repeats of `RepVGGBlock`                                 |
| `depth_mul`     | `float`                                                           | `0.33`                      | Depth multiplier                                                           |
| `width_mul`     | `float`                                                           | `0.25`                      | Width multiplier                                                           |
| `block`         | `Literal["RepBlock", "CSPStackRepBlock"]`                         | `"RepBlock"`                | Base block used                                                            |
| `csp_e`         | `float`                                                           | `0.5`                       | Factor for intermediate channels when block is set to `"CSPStackRepBlock"` |

### RexNetV1_lite

Adapted from [here](https://github.com/clovaai/rexnet)

**Parameters:**

| Key               | Type               | Default value    | Description                   |
| ----------------- | ------------------ | ---------------- | ----------------------------- |
| `fix_head_stem`   | `bool`             | `False`          | Whether to multiply head stem |
| `divisible_value` | `int`              | `8`              | Divisor used                  |
| `input_ch`        | `int`              | `16`             | tarting channel dimension     |
| `final_ch`        | `int`              | `164`            | Final channel dimension       |
| `multiplier`      | `float`            | `1.0`            | Channel dimension multiplier  |
| `kernel_sizes`    | `int \| list[int]` | `3`              | Kernel sizes                  |
| `out_indices`     | `list[int]`        | `[1, 4, 10, 17]` | Indices of the output layers  |

### `MobileOne`

Adapted from [here](https://github.com/apple/ml-mobileone).

**Parameters:**

| Key                | Type                                        | Default value | Description                                                                                     |
| ------------------ | ------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `variant`          | `Literal["s0", "s1", "s2", "s3", "s4"]`     | `"s0"`        | Variant of the network                                                                          |
| `width_multiplier` | `tuple[float, float, float, float] \| None` | `None`        | Width multiplier for each stage. If provided, overrides the variant value                       |
| `n_conv_branches`  | `int \| None`                               | `None`        | Number of convolutional branches in `MobileOne` block. If provided, overrides the variant value |
| `use_se`           | `bool \| None`                              | `None`        | Whether to use `Squeeze-and-Excitation` blocks. If provided, overrides the variant value        |

### `MobileNetV2`

Adapted from [here](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html).

**Parameters:**

| Key                | Type        | Default value    | Description                            |
| ------------------ | ----------- | ---------------- | -------------------------------------- |
| `download_weights` | `bool`      | `False`          | If True download weights from ImageNet |
| `out_indices`      | `list[int]` | `[3, 6, 13, 18]` | Indices of the output layers           |

### `EfficientNet`

Adapted from [here](https://github.com/rwightman/gen-efficientnet-pytorch).

**Parameters:**

| Key                | Type        | Default value     | Description                            |
| ------------------ | ----------- | ----------------- | -------------------------------------- |
| `download_weights` | `bool`      | `False`           | If True download weights from ImageNet |
| `out_indices`      | `list[int]` | `[0, 1, 2, 4, 6]` | Indices of the output layers           |

### `ContextSpatial`

Adapted from [here](https://github.com/taveraantonio/BiseNetv1).

**Parameters:**

| Key                | Type   | Default value   | Description                                                                                          |
| ------------------ | ------ | --------------- | ---------------------------------------------------------------------------------------------------- |
| `context_backbone` | `str`  | `"MobileNetV2"` | Backbone used for the context path. Must be a reference to a node registered in the `NODES` registry |
| `backbone_kwargs`  | `dict` | `{}`            | Keyword arguments for the context backbone                                                           |

### `DDRNet`

Adapted from [here](https://github.com/ydhongHIT/DDRNet)
**Parameters:**

| Key                           | Type                       | Default value       | Description                                                  |
| ----------------------------- | -------------------------- | ------------------- | ------------------------------------------------------------ |
| `variant`                     | `Literal["23-slim", "23"]` | `"23-slim"`         | Variant of the network                                       |
| `channels`                    | `int \| None`              | `None`              | Number of channels in the first layer                        |
| `highres_channels`            | `int \| None`              | `None`              | Number of channels in the high resolution branch             |
| `use_aux_heads`               | `bool`                     | `True`              | Whether to use auxiliary heads                               |
| `spp_width`                   | `int`                      | `128`               | Width of the spatial pyramid pooling layer                   |
| `spp_inter_mode`              | `str`                      | `"bilinear"`        | Up-sampling method for the spatial pyramid pooling layer     |
| `spp_kernel_sizes`            | `list[int]`                | `[1, 5, 9, 17, 0]`  | Kernel sizes for the spatial pyramid pooling layer           |
| `spp_strides`                 | `list[int]`                | `[1, 2, 4, 8, 0]`   | Strides for the spatial pyramid pooling layer                |
| `segmentation_inter_mode`     | `str`                      | `"bilinear"`        | Up-sampling method for the segmentation head                 |
| `layer5_bottleneck_expansion` | `int`                      | `2`                 | Expansion factor for the bottleneck layer in the fifth stage |
| `layer3_repeats`              | `int`                      | `3`                 | Number of repeats in the third stage                         |
| `layers`                      | `list[int]`                | `[2,2,2,2,1,2,2,1]` | Number of layers in each stage                               |

## Neck

### `RepPANNeck`

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key             | Type                                                              | Default value                    | Description                                                                     |
| --------------- | ----------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `variant`       | `Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]` | `"nano"`                         | Variant of the network                                                          |
| `n_heads`       | `Literal[2,3,4]`                                                  | `3`                              | Number of output heads. Should be same also on the connected head in most cases |
| `channels_list` | `list[int]`                                                       | `[256, 128, 128, 256, 256, 512]` | List of number of channels for each block                                       |
| `n_repeats`     | `list[int]`                                                       | `[12, 12, 12, 12]`               | List of number of repeats of `RepVGGBlock`                                      |
| `depth_mul`     | `float`                                                           | `0.33`                           | Depth multiplier                                                                |
| `width_mul`     | `float`                                                           | `0.25`                           | Width multiplier                                                                |
| `block`         | `Literal["RepBlock", "CSPStackRepBlock"]`                         | `"RepBlock"`                     | Base block used                                                                 |
| `csp_e`         | `float`                                                           | `0.5`                            | Factor for intermediate channels when block is set to `"CSPStackRepBlock"`      |

## Heads

### `ClassificationHead`

**Parameters:**

| Key            | Type    | Default value | Description                                      |
| -------------- | ------- | ------------- | ------------------------------------------------ |
| `dropout_rate` | `float` | `0.2`         | Dropout rate before last layer, range $\[0, 1\]$ |

### `SegmentationHead`

Adapted from [here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py).

### `BiSeNetHead`

Adapted from [here](https://github.com/taveraantonio/BiseNetv1).

**Parameters:**

| Key                     | Type  | Default value | Description                           |
| ----------------------- | ----- | ------------- | ------------------------------------- |
| `intermediate_channels` | `int` | `64`          | How many intermediate channels to use |

### `EfficientBBoxHead`

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key          | Type    | Default value | Description                                                                             |
| ------------ | ------- | ------------- | --------------------------------------------------------------------------------------- |
| `n_heads`    | `bool`  | `3`           | Number of output heads                                                                  |
| `conf_thres` | `float` | `0.25`        | Confidence threshold for non-maxima-suppression (used for evaluation)                   |
| `iou_thres`  | `float` | `0.45`        | `IoU` threshold for non-maxima-suppression (used for evaluation)                        |
| `max_det`    | `int`   | `300`         | Maximum number of detections retained after non-maxima-suppresion (used for evaluation) |

### `EfficientKeypointBBoxHead`

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf).

**Parameters:**

| Key           | Type           | Default value | Description                                                                             |
| ------------- | -------------- | ------------- | --------------------------------------------------------------------------------------- |
| `n_keypoints` | `int \| None ` | `None`        | Number of keypoints                                                                     |
| `n_heads`     | `int`          | `3`           | Number of output heads                                                                  |
| `conf_thres`  | `float`        | `0.25`        | Confidence threshold for non-maxima-suppression (used for evaluation)                   |
| `iou_thres`   | `float`        | `0.45`        | `IoU` threshold for non-maxima-suppression (used for evaluation)                        |
| `max_det`     | `int`          | `300`         | Maximum number of detections retained after non-maxima-suppresion (used for evaluation) |

### `DDRNetSegmentationHead`

Adapted from [here](https://github.com/ydhongHIT/DDRNet).

**Parameters:**

| Key              | Type  | Default value | Description                                                                                                               |
| ---------------- | ----- | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `inter_channels` | `int` | `64`          | Width of internal convolutions                                                                                            |
| `inter_mode`     | `str` | `"bilinear"`  | Up-sampling method. One of `"nearest"`, `"linear"`, `"bilinear"`, `"bicubic"`, `"trilinear"`, `"area"`, `"pixel_shuffle"` |
