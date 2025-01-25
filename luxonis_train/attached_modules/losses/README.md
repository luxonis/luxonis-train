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
- [Embedding Losses](#embedding-losses)

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
| `vis_kpts_loss_weight`  | `float`                                           | `1.0`         | Weight used for the keypoint visibility sub-loss                                                              |
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
| `object_weight` | `float` | `1000`        | Weight for the objects in the loss calculation. Training with a larger `object_weight` in the loss parameters may result in more false positives (FP), but it will improve accuracy. |

## Embedding Losses

We support the following losses taken from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/):

- [AngularLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#angularloss)
- [CircleLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#circleloss)
- [ContrastiveLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#contrastiveloss)
- [DynamicSoftMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#dynamicsoftmarginloss)
- [FastAPLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#fastaploss)
- [HistogramLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#histogramloss)
- [InstanceLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#instanceloss)
- [IntraPairVarianceLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#intrapairvarianceloss)
- [GeneralizedLiftedStructureLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#generalizedliftedstructureloss)
- [LiftedStructureLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#liftedstructureloss)
- [MarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#marginloss)
- [MultiSimilarityLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#multisimilarityloss)
- [NPairsLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#npairsloss)
- [NCALoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ncaloss)
- [NTXentLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#ntxentloss)
- [PNPLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#pnploss)
- [RankedListLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#rankedlistloss)
- [SignalToNoiseRatioContrastiveLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#signaltonoisecontrastiveloss)
- [SupConLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#supconloss)
- [ThresholdConsistentMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#thresholdconsistentmarginloss)
- [TripletMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss)
- [TupletMarginLoss](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tupletmarginloss)

**Parameters:**

For loss specific parameters, see the documentation pages linked above. In addition to the loss specific parameters, the following parameters are available:

| Key                  | Type   | Default value | Description                                                                                                                                                                                                                     |
| -------------------- | ------ | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `miner`              | `str`  | `None`        | Name of the miner to use with the loss. If `None`, no miner is used. All miners from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/) are supported.                                  |
| `miner_params`       | `dict` | `None`        | Parameters for the miner.                                                                                                                                                                                                       |
| `distance`           | `str`  | `None`        | Name of the distance metric to use with the loss. If `None`, no distance metric is used. All distance metrics from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/) are supported. |
| `distance_params`    | `dict` | `None`        | Parameters for the distance metric.                                                                                                                                                                                             |
| `reducer`            | `str`  | `None`        | Name of the reducer to use with the loss. If `None`, no reducer is used. All reducers from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/) are supported.                          |
| `reducer_params`     | `dict` | `None`        | Parameters for the reducer.                                                                                                                                                                                                     |
| `regularizer`        | `str`  | `None`        | Name of the regularizer to use with the loss. If `None`, no regularizer is used. All regularizers from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/) are supported.          |
| `regularizer_params` | `dict` | `None`        | Parameters for the regularizer.                                                                                                                                                                                                 |
