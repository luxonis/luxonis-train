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

| Key         | Type                             | Default value | Description                                                                    |
| ----------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------ |
| `alpha`     | `float \| list`                  | `0.25`        | Either a float for all channels or list of alphas for each channel             |
| `gamma`     | `float`                          | `2.0`         | Exponent of the modulating factor $(1 - p_t)$ to balance easy vs hard examples |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output                                 |

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
| `vis_kpts_loss_weight`  | `float`                                           | `1.0`         | Weight used for the keypoint visibility sub-loss                                                              |
| `sigmas`                | `list[float] \ None`                              | `None`        | Sigmas used in `KeypointLoss` for `OKS` metric. If `None` then use COCO ones if possible or default ones      |
| `area_factor`           | `float \| None`                                   | `None`        | Factor by which we multiply bounding box area which is used in `KeypointLoss.` If `None` then use default one |
