# Nodes

Nodes are the basic building structures of the model. They can be connected together
arbitrarily as long as the two nodes are compatible with each other.

## Table Of Contents

- [ResNet18](#resnet18)
- [MicroNet](#micronet)
- [RepVGG](#repvgg)
- [EfficientRep](#efficientrep)
- [RexNetV1_lite](#rexnetv1_lite)
- [MobileOne](#mobileone)
- [MobileNetV2](#mobilenetv2)
- [EfficientNet](#efficientnet)
- [ContextSpatial](#contextspatial)
- [RepPANNeck](#reppanneck)
- [ClassificationHead](#classificationhead)
- [SegmentationHead](#segmentationhead)
- [BiSeNetHead](#bisenethead)
- [EfficientBBoxHead](#efficientbboxhead)
- [ImplicitKeypointBBoxHead](#implicitkeypointbboxhead)

Every node takes these parameters:

| Key          | Type        | Default value | Description                                                                                                               |
| ------------ | ----------- | ------------- | ------------------------------------------------------------------------------------------------------------------------- |
| attach_index | int \| None | None          | Index of previous output that the head attaches to. Each node has a sensible default. Usually should not be manually set. |
| n_classes    | int \| None | None          | Number of classes in the dataset. Inferred from the dataset if not provided.                                              |

Additional parameters for specific nodes are listed below.

## ResNet18

Adapted from [here](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html).

**Params**

| Key              | Type | Default value | Description                            |
| ---------------- | ---- | ------------- | -------------------------------------- |
| download_weights | bool | False         | If True download weights from imagenet |

## MicroNet

Adapted from [here](https://github.com/liyunsheng13/micronet).

**Params**

| Key     | Type                        | Default value | Description             |
| ------- | --------------------------- | ------------- | ----------------------- |
| variant | Literal\["M1", "M2", "M3"\] | "M1"          | Variant of the network. |

## RepVGG

Adapted from [here](https://github.com/DingXiaoH/RepVGG).

**Params**

| Key     | Type                        | Default value | Description             |
| ------- | --------------------------- | ------------- | ----------------------- |
| variant | Literal\["A0", "A1", "A2"\] | "A0"          | Variant of the network. |

## EfficientRep

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Params**

| Key           | Type        | Default value               | Description                                         |
| ------------- | ----------- | --------------------------- | --------------------------------------------------- |
| channels_list | List\[int\] | \[64, 128, 256, 512, 1024\] | List of number of channels for each block           |
| num_repeats   | List\[int\] | \[1, 6, 12, 18, 6\]         | List of number of repeats of RepVGGBlock            |
| in_channels   | int         | 3                           | Number of input channels, should be 3 in most cases |
| depth_mul     | int         | 0.33                        | Depth multiplier                                    |
| width_mul     | int         | 0.25                        | Width multiplier                                    |

## RexNetV1_lite

Adapted from ([here](https://github.com/clovaai/rexnet).

**Params**

| Key             | Type  | Default value | Description                    |
| --------------- | ----- | ------------- | ------------------------------ |
| fix_head_stem   | bool  | False         | Whether to multiply head stem  |
| divisible_value | int   | 8             | Divisor used                   |
| input_ch        | int   | 16            | tarting channel dimension      |
| final_ch        | int   | 164           | Final channel dimension        |
| multiplier      | float | 1.0           | Channel dimension multiplier   |
| kernel_conf     | str   | '333333'      | Kernel sizes encoded as string |

## MobileOne

Adapted from [here](https://github.com/apple/ml-mobileone).

**Params**

| Key     | Type                                    | Default value | Description             |
| ------- | --------------------------------------- | ------------- | ----------------------- |
| variant | Literal\["s0", "s1", "s2", "s3", "s4"\] | "s0"          | Variant of the network. |

## MobileNetV2

Adapted from [here](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html).

**Params**

| Key              | Type | Default value | Description                            |
| ---------------- | ---- | ------------- | -------------------------------------- |
| download_weights | bool | False         | If True download weights from imagenet |

## EfficientNet

Adapted from [here](https://github.com/rwightman/gen-efficientnet-pytorch).

**Params**

| Key              | Type | Default value | Description                             |
| ---------------- | ---- | ------------- | --------------------------------------- |
| download_weights | bool | False         | If True download weights from imagenet. |

## ContextSpatial

Adapted from [here](https://github.com/taveraantonio/BiseNetv1).

**Params**

| Key              | Type | Default value | Description   |
| ---------------- | ---- | ------------- | ------------- |
| context_backbone | str  | "MobileNetV2" | Backbone used |

## RepPANNeck

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Params**

| Key           | Type             | Default value                                           | Description                               |
| ------------- | ---------------- | ------------------------------------------------------- | ----------------------------------------- |
| num_heads     | Literal\[2,3,4\] | 3 ***Note:** Should be same also on head in most cases* | Number of output heads                    |
| channels_list | List\[int\]      | \[256, 128, 128, 256, 256, 512\]                        | List of number of channels for each block |
| num_repeats   | List\[int\]      | \[12, 12, 12, 12\]                                      | List of number of repeats of RepVGGBlock  |
| depth_mul     | int              | 0.33                                                    | Depth multiplier                          |
| width_mul     | int              | 0.25                                                    | Width multiplier                          |

## ClassificationHead

**Params**

| Key        | Type  | Default value | Description                                   |
| ---------- | ----- | ------------- | --------------------------------------------- |
| fc_dropout | float | 0.2           | Dropout rate before last layer, range \[0,1\] |

## SegmentationHead

Adapted from [here](https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/fcn.py).

## BiSeNetHead

Adapted from [here](https://github.com/taveraantonio/BiseNetv1).

**Params**

| Key            | Type | Default value | Description                                    |
| -------------- | ---- | ------------- | ---------------------------------------------- |
| upscale_factor | int  | 8             | Factor used for upscaling input                |
| is_aux         | bool | False         | Either use 256 for intermediate channels or 64 |

## EfficientBBoxHead

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Params**

| Key       | Type | Default value | Description            |
| --------- | ---- | ------------- | ---------------------- |
| num_heads | bool | 3             | Number of output heads |

## ImplicitKeypointBBoxHead

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf).

**Params**

| Key              | Type                        | Default value | Description                                                                                                |
| ---------------- | --------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| n_keypoints      | int \| None                 | None          | Number of keypoints.                                                                                       |
| num_heads        | int                         | 3             | Number of output heads                                                                                     |
| anchors          | List\[List\[int\]\] \| None | None          | Anchors used for object detection. If set to `None`, the anchors are computed at runtime from the dataset. |
| init_coco_biases | bool                        | True          | Whether to use COCO bias and weight initialization                                                         |
| conf_thres       | float                       | 0.25          | confidence threshold for nms (used for evaluation)                                                         |
| iou_thres        | float                       | 0.45          | iou threshold for nms (used for evaluation)                                                                |
