# Losses

List of all the available loss functions.

## Table Of Contents

- [CrossEntropyLoss](#crossentropyloss)
- [BCEWithLogitsLoss](#bcewithlogitsloss)
- [SmoothBCEWithLogitsLoss](#smoothbcewithlogitsloss)
- [SigmoidFocalLoss](#sigmoidfocalloss)
- [SoftmaxFocalLoss](#softmaxfocalloss)
- [AdaptiveDetectionLoss](#adaptivedetectionloss)
- [ImplicitKeypointBBoxLoss](#implicitkeypointbboxloss)

## CrossEntropyLoss

Adapted from [here](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

**Params**

| Key             | Type                             | Default value | Description                                                                                                                                                                                                                                                                                  |
| --------------- | -------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| weight          | list\[float\] \| None            | None          | A manual rescaling weight given to each class. If given, it has to be a list of the same length as there are classes.                                                                                                                                                                        |
| reduction       | Literal\["none", "mean", "sum"\] | "mean"        | Specifies the reduction to apply to the output.                                                                                                                                                                                                                                              |
| label_smoothing | float\[0.0, 1.0\]                | 0.0           | Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). |

## BCEWithLogitsLoss

Adapted from [here](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

**Params**

| Key          | Type                             | Default value | Description                                                                                                                                                                                                                                               |
| ------------ | -------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| weight       | list\[float\] \| None            | None          | A manual rescaling weight given to each class. If given, has to be a list of the same length as there are classes.                                                                                                                                        |
| ignore_index | int                              | -100          | Specifies a target value that is ignored and does not contribute to the input gradient. When `size_average` is `True`, the loss is averaged over non-ignored targets. Note that `ignore_index` is only applicable when the target contains class indices. |
| reduction    | Literal\["none", "mean", "sum"\] | "mean"        | Specifies the reduction to apply to the output.                                                                                                                                                                                                           |

## SmoothBCEWithLogitsLoss

**Params**

| Key             | Type                             | Default value | Description                                                                                                                                                                                                                                                                                  |
| --------------- | -------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| weight          | list\[float\] \| None            | None          | A manual rescaling weight given to each class. If given, has to be a list of the same length as there are classes.                                                                                                                                                                           |
| reduction       | Literal\["none", "mean", "sum"\] | "mean"        | Specifies the reduction to apply to the output.                                                                                                                                                                                                                                              |
| label_smoothing | float\[0.0, 1.0\]                | 0.0           | Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets become a mixture of the original ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). |
| bce_pow         | float                            | 1.0           | Weight for the positive samples.                                                                                                                                                                                                                                                             |

## SigmoidFocalLoss

Adapted from [here](https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html#torchvision.ops.sigmoid_focal_loss).

**Params**

| Key       | Type                             | Default value | Description                                                                                |
| --------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| alpha     | float                            | 0.25          | Weighting factor in range (0,1) to balance positive vs negative examples or -1 for ignore. |
| gamma     | float                            | 2.0           | Exponent of the modulating factor $(1 - p_t)$ to balance easy vs hard examples             |
| reduction | Literal\["none", "mean", "sum"\] | "mean"        | Specifies the reduction to apply to the output.                                            |

## SoftmaxFocalLoss

**Params**

| Key       | Type                             | Default value | Description                                                                   |
| --------- | -------------------------------- | ------------- | ----------------------------------------------------------------------------- |
| alpha     | float \| list                    | 0.25          | Either a float for all channels or list of alphas for each channel.           |
| gamma     | float                            | 2.0           | Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples. |
| reduction | Literal\["none", "mean", "sum"\] | "mean"        | Specifies the reduction to apply to the output.                               |

## AdaptiveDetectionLoss

Adapted from [here](https://arxiv.org/pdf/2209.02976.pdf).

**Params**

| Key               | Type                                              | Default value | Description                                                                         |
| ----------------- | ------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| n_warmup_epochs   | int                                               | 4             | Number of epochs where ATSS assigner is used, after that we switch to TAL assigner. |
| iou_type          | Literal\["none", "giou", "diou", "ciou", "siou"\] | "giou"        | IoU type used for bbox regression loss.                                             |
| class_loss_weight | float                                             | 1.0           | Weight used for the classification part of the loss.                                |
| iou_loss_weight   | float                                             | 2.5           | Weight used for the IoU part of the loss.                                           |

## ImplicitKeypointBBoxLoss

Adapted from [YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object
Keypoint Similarity Loss](https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf).

**Params**

| Key                             | Type          | Default value     | Description                                                                                |
| ------------------------------- | ------------- | ----------------- | ------------------------------------------------------------------------------------------ |
| cls_pw                          | float         | 1.0               | Power for the [SmoothBCEWithLogitsLoss](#smoothbcewithlogitsloss) for classification loss. |
| obj_pw                          | float         | 1.0               | Power for [BCEWithLogitsLoss](#bcewithlogitsloss) for objectness loss.                     |
| viz_pw                          | float         | 1.0               | Power for [BCEWithLogitsLoss](#bcewithlogitsloss) for keypoint visibility.                 |
| label_smoothing                 | float         | 0.0               | Smoothing for [SmothBCEWithLogitsLoss](#smoothbcewithlogitsloss) for classification loss.  |
| min_objectness_iou              | float         | 0.0               | Minimum objectness IoU.                                                                    |
| bbox_loss_weight                | float         | 0.05              | Weight for bbox detection sub-loss.                                                        |
| keypoint_distance_loss_weight   | float         | 0.10              | Weight for keypoint distance sub-loss.                                                     |
| keypoint_visibility_loss_weight | float         | 0.6               | Weight for keypoint visibility sub-loss.                                                   |
| class_loss_weight               | float         | 0.6               | Weight for classification sub-loss.                                                        |
| objectness_loss_weight          | float         | 0.7               | Weight for objectness sub-loss.                                                            |
| anchor_threshold                | float         | 4.0               | Threshold for matching anchors to targets.                                                 |
| bias                            | float         | 0.5               | Bias for matchinf anchors to targets.                                                      |
| balance                         | list\[float\] | \[4.0, 1.0, 0.4\] | Balance for objectness loss.                                                               |
