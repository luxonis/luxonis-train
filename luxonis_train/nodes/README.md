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
  - [`RecSubNet`](#recsubnet)
  - [`EfficientViT`](#efficientvit)
  - [`GhostFaceNetV2`](#ghostfacenetv2)
- [Necks](#necks)
  - [`RepPANNeck`](#reppanneck)
- [Heads](#heads)
  - [`ClassificationHead`](#classificationhead)
  - [`SegmentationHead`](#segmentationhead)
  - [`BiSeNetHead`](#bisenethead)
  - [`EfficientBBoxHead`](#efficientbboxhead)
  - [`EfficientKeypointBBoxHead`](#efficientkeypointbboxhead)
  - [`DDRNetSegmentationHead`](#ddrnetsegmentationhead)
  - [`DiscSubNetHead`](#discsubnet)
  - [`FOMOHead`](#fomohead)
  - [`GhostFaceNetHead`](#ghostfacenethead)
  - [`PrecisionBBoxHead`](#precisionbboxhead)
  - [`PrecisionSegmentBBoxHead`](#precisionsegmentbboxhead)

Every node takes these parameters:

| Key                | Type          | Default value | Description                                                                 |
| ------------------ | ------------- | ------------- | --------------------------------------------------------------------------- |
| `n_classes`        | `int \| None` | `None`        | Number of classes in the dataset. Inferred from the dataset if not provided |
| `remove_on_export` | `bool`        | `False`       | Whether node should be removed when exporting the whole model               |

In addition, the following class attributes can be overridden:

| Key            | Type                                                              | Default value | Description                                                                                                                                                                                                                     |
| -------------- | ----------------------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attach_index` | `int \| "all" \| tuple[int, int] \| tuple[int, int, int] \| None` | `None`        | Index of previous output that the head attaches to. Each node has a sensible default. Usually should not be manually set in most cases. Can be either a single index, a slice (negative indexing is also supported), or `"all"` |
| `task`         | `Task \| None`                                                    | `None`        | List of tasks types supported by the node. Should be overridden for head nodes.                                                                                                                                                 |

Additional parameters for specific nodes are listed below.

## Backbones

### `ResNet`

Adapted from [here](https://pytorch.org/vision/main/models/resnet.html).

**Parameters:**

| Key                | Type                                      | Default value | Description                            |
| ------------------ | ----------------------------------------- | ------------- | -------------------------------------- |
| `variant`          | `Literal["18", "34", "50", "101", "152"]` | `"18"`        | Variant of the network                 |
| `download_weights` | `bool`                                    | `True`        | If True download weights from ImageNet |

### `MicroNet`

Adapted from [here](https://github.com/liyunsheng13/micronet).

**Parameters:**

| Key       | Type                        | Default value | Description            |
| --------- | --------------------------- | ------------- | ---------------------- |
| `variant` | `Literal["M1", "M2", "M3"]` | `"M1"`        | Variant of the network |

### `RepVGG`

Adapted from [here](https://github.com/DingXiaoH/RepVGG).

**Parameters:**

| Key       | Type                        | Default value | Description            |
| --------- | --------------------------- | ------------- | ---------------------- |
| `variant` | `Literal["A0", "A1", "A2"]` | `"A0"`        | Variant of the network |

### `EfficientRep`

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key                  | Type                                                              | Default value               | Description                                                                |
| -------------------- | ----------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------- |
| `variant`            | `Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]` | `"nano"`                    | Variant of the network                                                     |
| `channels_list`      | `list[int]`                                                       | \[64, 128, 256, 512, 1024\] | List of number of channels for each block                                  |
| `n_repeats`          | `list[int]`                                                       | \[1, 6, 12, 18, 6\]         | List of number of repeats of `RepVGGBlock`                                 |
| `depth_mul`          | `float`                                                           | `0.33`                      | Depth multiplier                                                           |
| `width_mul`          | `float`                                                           | `0.25`                      | Width multiplier                                                           |
| `block`              | `Literal["RepBlock", "CSPStackRepBlock"]`                         | `"RepBlock"`                | Base block used                                                            |
| `csp_e`              | `float`                                                           | `0.5`                       | Factor for intermediate channels when block is set to `"CSPStackRepBlock"` |
| `download_weights`   | `bool`                                                            | `True`                      | If True download weights from COCO (if available for specified variant)    |
| `initialize_weights` | `bool`                                                            | `True`                      | If True, initialize weights.                                               |

### RexNetV1_lite

Adapted from [here](https://github.com/clovaai/rexnet)

**Parameters:**

| Key               | Type               | Default value | Description                   |
| ----------------- | ------------------ | ------------- | ----------------------------- |
| `fix_head_stem`   | `bool`             | `False`       | Whether to multiply head stem |
| `divisible_value` | `int`              | `8`           | Divisor used                  |
| `input_ch`        | `int`              | `16`          | tarting channel dimension     |
| `final_ch`        | `int`              | `164`         | Final channel dimension       |
| `multiplier`      | `float`            | `1.0`         | Channel dimension multiplier  |
| `kernel_sizes`    | `int \| list[int]` | `3`           | Kernel sizes                  |

### `MobileOne`

Adapted from [here](https://github.com/apple/ml-mobileone).

**Parameters:**

| Key       | Type                                    | Default value | Description            |
| --------- | --------------------------------------- | ------------- | ---------------------- |
| `variant` | `Literal["s0", "s1", "s2", "s3", "s4"]` | `"s0"`        | Variant of the network |

### `MobileNetV2`

Adapted from [here](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html).

**Parameters:**

| Key                | Type   | Default value | Description                            |
| ------------------ | ------ | ------------- | -------------------------------------- |
| `download_weights` | `bool` | `True`        | If True download weights from ImageNet |

### `EfficientNet`

Adapted from [here](https://github.com/rwightman/gen-efficientnet-pytorch).

**Parameters:**

| Key                | Type   | Default value | Description                            |
| ------------------ | ------ | ------------- | -------------------------------------- |
| `download_weights` | `bool` | `True`        | If True download weights from ImageNet |

### `ContextSpatial`

Adapted from [here](https://github.com/taveraantonio/BiseNetv1).

**Parameters:**

| Key                | Type  | Default value   | Description                                                                                          |
| ------------------ | ----- | --------------- | ---------------------------------------------------------------------------------------------------- |
| `context_backbone` | `str` | `"MobileNetV2"` | Backbone used for the context path. Must be a reference to a node registered in the `NODES` registry |

### `DDRNet`

Adapted from [here](https://github.com/ydhongHIT/DDRNet)

**Parameters:**

| Key                | Type                       | Default value | Description                                                             |
| ------------------ | -------------------------- | ------------- | ----------------------------------------------------------------------- |
| `variant`          | `Literal["23-slim", "23"]` | `"23-slim"`   | Variant of the network                                                  |
| `download_weights` | `bool`                     | `True`        | If True download weights from COCO (if available for specified variant) |

### `RecSubNet`

Adapted from [here](https://arxiv.org/abs/2108.07610)

**Parameters:**

| Key       | Type                | Default value | Description            |
| --------- | ------------------- | ------------- | ---------------------- |
| `variant` | `Literal["n", "l"]` | `"l"`         | Variant of the network |

### `EfficientViT`

Adapted from [here](https://arxiv.org/abs/2205.14756)

**Parameters:**

| Key            | Type                                                              | Default value                    | Description                                         |
| -------------- | ----------------------------------------------------------------- | -------------------------------- | --------------------------------------------------- |
| `variant`      | `Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]` | `"nano"`                         | Variant of the network                              |
| `width_list`   | `list[int]`                                                       | `[256, 256, 256, 256, 256, 512]` | List of number of channels for each block           |
| `depth_list`   | `list[int]`                                                       | `[12, 12, 12, 12]`               | List of number of repeats of `EfficientViTBlock`    |
| `expand_ratio` | `int`                                                             | `4`                              | Factor by which channels expand in the local module |
| `dim`          | `int`                                                             | `None`                           | Dimension size for each attention head              |

### `GhostFaceNetV2`

**Parameters:**

| Key       | Type            | Default value | Description                 |
| --------- | --------------- | ------------- | --------------------------- |
| `variant` | `Literal["V2"]` | `"V2"`        | The variant of the network. |

## Neck

### `RepPANNeck`

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key                  | Type                                                              | Default value                    | Description                                                                     |
| -------------------- | ----------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------- |
| `variant`            | `Literal["n", "nano", "s", "small", "m", "medium", "l", "large"]` | `"nano"`                         | Variant of the network                                                          |
| `n_heads`            | `Literal[2,3,4]`                                                  | `3`                              | Number of output heads. Should be same also on the connected head in most cases |
| `channels_list`      | `list[int]`                                                       | `[256, 128, 128, 256, 256, 512]` | List of number of channels for each block                                       |
| `n_repeats`          | `list[int]`                                                       | `[12, 12, 12, 12]`               | List of number of repeats of `RepVGGBlock`                                      |
| `depth_mul`          | `float`                                                           | `0.33`                           | Depth multiplier                                                                |
| `width_mul`          | `float`                                                           | `0.25`                           | Width multiplier                                                                |
| `block`              | `Literal["RepBlock", "CSPStackRepBlock"]`                         | `"RepBlock"`                     | Base block used                                                                 |
| `csp_e`              | `float`                                                           | `0.5`                            | Factor for intermediate channels when block is set to `"CSPStackRepBlock"`      |
| `download_weights`   | `bool`                                                            | `False`                          | If True download weights from COCO (if available for specified variant)         |
| `initialize_weights` | `bool`                                                            | `True`                           | If True, initialize weights.                                                    |

## Heads

### `ClassificationHead`

**Parameters:**

| Key          | Type    | Default value | Description                                      |
| ------------ | ------- | ------------- | ------------------------------------------------ |
| `fc_dropout` | `float` | `0.2`         | Dropout rate before last layer, range $\[0, 1\]$ |

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

| Key                  | Type    | Default value | Description                                                           |
| -------------------- | ------- | ------------- | --------------------------------------------------------------------- |
| `n_heads`            | `int`   | `3`           | Number of output heads                                                |
| `conf_thres`         | `float` | `0.25`        | Confidence threshold for non-maxima-suppression (used for evaluation) |
| `iou_thres`          | `float` | `0.45`        | `IoU` threshold for non-maxima-suppression (used for evaluation)      |
| `max_det`            | `int`   | `300`         | Maximum number of detections retained after NMS                       |
| `download_weights`   | `bool`  | `False`       | If True download weights from COCO                                    |
| `initialize_weights` | `bool`  | `True`        | If True, initialize weights.                                          |

### `EfficientKeypointBBoxHead`

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf).

**Parameters:**

| Key           | Type           | Default value | Description                                                           |
| ------------- | -------------- | ------------- | --------------------------------------------------------------------- |
| `n_keypoints` | `int \| None ` | `None`        | Number of keypoints                                                   |
| `n_heads`     | `int`          | `3`           | Number of output heads                                                |
| `conf_thres`  | `float`        | `0.25`        | Confidence threshold for non-maxima-suppression (used for evaluation) |
| `iou_thres`   | `float`        | `0.45`        | `IoU` threshold for non-maxima-suppression (used for evaluation)      |

### `DDRNetSegmentationHead`

Adapted from [here](https://github.com/ydhongHIT/DDRNet).

**Parameters:**

| Key                | Type   | Default value | Description                                                                                                               |
| ------------------ | ------ | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `inter_channels`   | `int`  | `64`          | Width of internal convolutions                                                                                            |
| `inter_mode`       | `str`  | `"bilinear"`  | Up-sampling method. One of `"nearest"`, `"linear"`, `"bilinear"`, `"bicubic"`, `"trilinear"`, `"area"`, `"pixel_shuffle"` |
| `download_weights` | `bool` | `False`       | If True download weights from COCO                                                                                        |

### `DiscSubNetHead`

Adapted from [here](https://arxiv.org/abs/2108.07610).

**Parameters:**

| Key       | Type                | Default value | Description            |
| --------- | ------------------- | ------------- | ---------------------- |
| `variant` | `Literal["n", "l"]` | `"l"`         | Variant of the network |

### `FOMOHead`

**Parameters:**

| Key               | Type   | Default value | Description                                                                              |
| ----------------- | ------ | ------------- | ---------------------------------------------------------------------------------------- |
| `num_conv_layers` | `int`  | `3`           | Number of convolutional layers to use in the model.                                      |
| `conv_channels`   | `int`  | `16`          | Number of output channels for each convolutional layer.                                  |
| `use_nms`         | `bool` | `False`       | If True, enable NMS. This can reduce FP, but it will also reduce TP for close neighbors. |

### `GhostFaceNetHead`

**Parameters:**

| Key              | Type  | Default value | Description                              |
| ---------------- | ----- | ------------- | ---------------------------------------- |
| `embedding_size` | `int` | `512`         | The size of the output embedding vector. |

### `PrecisionBBoxHead`

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf) and [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key          | Type    | Default value | Description                                                               |
| ------------ | ------- | ------------- | ------------------------------------------------------------------------- |
| `reg_max`    | `int`   | `16`          | Maximum number of regression channels                                     |
| `n_heads`    | `int`   | `3`           | Number of output heads                                                    |
| `conf_thres` | `float` | `0.25`        | Confidence threshold for non-maxima-suppression (used for evaluation)     |
| `iou_thres`  | `float` | `0.45`        | IoU threshold for non-maxima-suppression (used for evaluation)            |
| `max_det`    | `int`   | `300`         | Max number of detections for non-maxima-suppression (used for evaluation) |

### `PrecisionSegmentBBoxHead`

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf) and [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key          | Type    | Default value | Description                                                                |
| ------------ | ------- | ------------- | -------------------------------------------------------------------------- |
| `reg_max`    | `int`   | `16`          | Maximum number of regression channels.                                     |
| `n_heads`    | `int`   | `3`           | Number of output heads.                                                    |
| `conf_thres` | `float` | `0.25`        | Confidence threshold for non-maxima-suppression (used for evaluation).     |
| `iou_thres`  | `float` | `0.45`        | IoU threshold for non-maxima-suppression (used for evaluation).            |
| `max_det`    | `int`   | `300`         | Max number of detections for non-maxima-suppression (used for evaluation). |
| `n_masks`    | `int`   | `32`          | Number of of output instance segmentation masks at the output.             |
| `n_proto`    | `int`   | `256`         | Number of prototypes generated from the prototype generator.               |
