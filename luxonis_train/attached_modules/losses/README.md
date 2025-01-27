# Losses

List of all the available loss functions.

## Table Of Contents

- [`CrossEntropyLoss`](#crossentropyloss)
- [`BCEWithLogitsLoss`](#bcewithlogitsloss)
- [`SmoothBCEWithLogitsLoss`](#smoothbcewithlogitsloss)
- [`SigmoidFocalLoss`](#sigmoidfocalloss)
- [`SoftmaxFocalLoss`](#softmaxfocalloss)
- [`AdaptiveDetectionLoss`](#adaptivedetectionloss)
- [`EfficientKeypointBBoxLoss`](#efficientkeypointbboxloss)
- [`FOMOLocalizationLoss`](#fomolocalizationLoss)
- \[`PrecisionDFLDetectionLoss`\] (# precisiondfldetectionloss)
- \[`PrecisionDFLSegmentationLoss`\] (# precisiondflsegmentationloss)

## `CrossEntropyLoss`

Adapted from [here](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

**Parameters:**

| Key               | Type                             | Default value | Description                                                                                                                                                                                                                                                                                 |
| ----------------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `weight`          | `list[float] \| None`            | `None`        | A manual rescaling weight given to each class. If given, it has to be a list of the same length as there are classes                                                                                                                                                                        |
| `reduction`       | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output                                                                                                                                                                                                                                              |
| `label_smoothing` | `float` $\\in \[0.0, 1.0\]$      | `0.0`         | Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) |

## `BCEWithLogitsLoss`

Adapted from [here](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

**Parameters:**

| Key          | Type                             | Default value | Description                                                                                                       |
| ------------ | -------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------- |
| `weight`     | `list[float] \| None`            | `None`        | A manual rescaling weight given to each class. If given, has to be a list of the same length as there are classes |
| `reduction`  | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output                                                                    |
| `pos_weight` | `Tensor \| None`                 | `None`        | A weight of positive examples to be broadcasted with target                                                       |

## `SmoothBCEWithLogitsLoss`

**Parameters:**

| Key               | Type                             | Default value | Description                                                                                                                                                                                                                                                                                 |
| ----------------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `weight`          | `list[float] \| None`            | `None`        | A manual rescaling weight given to each class. If given, has to be a list of the same length as there are classes                                                                                                                                                                           |
| `reduction`       | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output                                                                                                                                                                                                                                              |
| `label_smoothing` | `float` $\\in \[0.0, 1.0\]$      | `0.0`         | Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) |
| `bce_pow`         | `float`                          | `1.0`         | Weight for the positive samples                                                                                                                                                                                                                                                             |

## `SigmoidFocalLoss`

Adapted from [here](https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html#torchvision.ops.sigmoid_focal_loss).

**Parameters:**

| Key         | Type                             | Default value | Description                                                                                 |
| ----------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------- |
| `alpha`     | `float`                          | `0.25`        | Weighting factor in range $(0,1)$ to balance positive vs negative examples or -1 for ignore |
| `gamma`     | `float`                          | `2.0`         | Exponent of the modulating factor $(1 - p_t)$ to balance easy vs hard examples              |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output                                              |

## `SoftmaxFocalLoss`

**Parameters:**

| Key         | Type                             | Default value | Description                                                                                                                                                  |
| ----------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `alpha`     | `float \| list[float]`           | `1`           | Balancing factor to reduce the impact of class imbalance. Can be a single value or a list of values per class.                                               |
| `gamma`     | `float`                          | `2.0`         | Exponent of the modulating factor $(1 - p_t)$ to balance easy vs hard examples                                                                               |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output                                                                                                               |
| `smooth`    | `float`                          | `1e-5`        | A small value added to labels to prevent zero probabilities, helping to smooth and stabilize training by making the model less confident in its predictions. |

## `AdaptiveDetectionLoss`

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key                 | Type                                              | Default value | Description                                                                            |
| ------------------- | ------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------- |
| `n_warmup_epochs`   | `int`                                             | `4`           | Number of epochs where `ATSS` assigner is used, after that we switch to `TAL` assigner |
| `iou_type`          | `Literal["none", "giou", "diou", "ciou", "siou"]` | `"giou"`      | `IoU` type used for bounding box regression loss                                       |
| `class_loss_weight` | `float`                                           | `1.0`         | Weight used for the classification part of the loss                                    |
| `iou_loss_weight`   | `float`                                           | `2.5`         | Weight used for the `IoU` part of the loss                                             |

## `EfficientKeypointBBoxLoss`

Adapted from [YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object
Keypoint Similarity Loss](https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf).

| Key                     | Type                                              | Default value | Description                                                                                                   |
| ----------------------- | ------------------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------- |
| `n_warmup_epochs`       | `int`                                             | `4`           | Number of epochs where `ATSS` assigner is used, after that we switch to `TAL` assigner                        |
| `iou_type`              | `Literal["none", "giou", "diou", "ciou", "siou"]` | `"giou"`      | `IoU` type used for bounding box regression sub-loss                                                          |
| `reduction`             | `Literal["mean", "sum"]`                          | `"mean"`      | Specifies the reduction to apply to the output                                                                |
| `class_loss_weight`     | `float`                                           | `1.0`         | Weight used for the classification sub-loss                                                                   |
| `iou_loss_weight`       | `float`                                           | `2.5`         | Weight used for the `IoU` sub-loss                                                                            |
| `regr_kpts_loss_weight` | `float`                                           | `1.5`         | Weight used for the `OKS` sub-loss                                                                            |
| `vis_kpts_loss_weight`  | `float`                                           | `2.0`         | Weight used for the keypoint visibility sub-loss                                                              |
| `sigmas`                | `list[float] \ None`                              | `None`        | Sigmas used in `KeypointLoss` for `OKS` metric. If `None` then use COCO ones if possible or default ones      |
| `area_factor`           | `float \| None`                                   | `None`        | Factor by which we multiply bounding box area which is used in `KeypointLoss.` If `None` then use default one |

## `ReconstructionSegmentationLoss`

Adapted from [here](https://arxiv.org/abs/2108.07610).

**Parameters:**

| Key         | Type                             | Default value | Description                                                                              |
| ----------- | -------------------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| `alpha`     | `float \| list`                  | `1.0`         | Weighting factor for Focal loss, either a single float or list of values for each class. |
| `gamma`     | `float`                          | `2.0`         | Modulates the balance between easy and hard examples in Focal loss.                      |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies how to reduce the output of the Focal loss.                                    |
| `smooth`    | `float`                          | `1e-5`        | Smoothing factor to prevent overconfidence in predictions for Focal loss.                |

## `FOMOLocalizationLoss`

**Parameters:**

| Key             | Type    | Default value | Description                                                                                                                                                                          |
| --------------- | ------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `object_weight` | `float` | `500`         | Weight for the objects in the loss calculation. Training with a larger `object_weight` in the loss parameters may result in more false positives (FP), but it will improve accuracy. |

## `CTCLoss`

CTC loss with optional focal loss weighting.

**Parameters:**

| Key              | Type   | Default value | Description                                            |
| ---------------- | ------ | ------------- | ------------------------------------------------------ |
| `use_focal_loss` | `bool` | `True`        | Whether to apply focal loss weighting to the CTC loss. |

## `PrecisionDFLDetectionLoss`

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf) and [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key                 | Type    | Default value | Description                                |
| ------------------- | ------- | ------------- | ------------------------------------------ |
| `tal_topk`          | `int`   | `10`          | Number of anchors considered in selection. |
| `class_loss_weight` | `float` | `0.5`         | Weight for classification loss.            |
| `bbox_loss_weight`  | `float` | `7.5`         | Weight for bbox loss.                      |
| `dfl_loss_weigth`   | `float` | `1.5`         | Weight for DFL loss.                       |

## `PrecisionDFLSegmentationLoss`

Adapted from [here](https://arxiv.org/pdf/2207.02696.pdf) and [here](https://arxiv.org/pdf/2209.02976.pdf).

**Parameters:**

| Key                 | Type    | Default value | Description                                |
| ------------------- | ------- | ------------- | ------------------------------------------ |
| `tal_topk`          | `int`   | `10`          | Number of anchors considered in selection. |
| `class_loss_weight` | `float` | `0.5`         | Weight for classification loss.            |
| `bbox_loss_weight`  | `float` | `7.5`         | Weight for bbox and segmentation loss.     |
| `dfl_loss_weigth`   | `float` | `1.5`         | Weight for DFL loss.                       |
