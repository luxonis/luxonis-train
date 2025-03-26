# Losses

Loss functions are organized based on the tasks they serve. This helps in quickly locating the loss that best suits your application—whether it be classification, detection, anomaly detection, OCR, or embedding learning.

## Table Of Contents

- [Classification & Segmentation Losses](#classification--segmentation-losses)
  - [`CrossEntropyLoss`](#crossentropyloss)
  - [`BCEWithLogitsLoss`](#bcewithlogitsloss)
  - [`SmoothBCEWithLogitsLoss`](#smoothbcewithlogitsloss)
  - [`SigmoidFocalLoss`](#sigmoidfocalloss)
  - [`SoftmaxFocalLoss`](#softmaxfocalloss)
  - [`OHEMCrossEntropyLoss`](#ohemcrossentropyloss)
  - [`OHEMBCEWithLogitsLoss`](#ohembcewithlogitsloss)
- [Bounding Box Detection Losses](#bounding-box-detection-losses)
  - [`AdaptiveDetectionLoss`](#adaptivedetectionloss)
  - [`PrecisionDFLDetectionLoss`](#precisiondfldetectionloss)
- [Instance Keypoint Detection Losses](#instance-keypoint-detection-losses)
  - [`EfficientKeypointBBoxLoss`](#efficientkeypointbboxloss)
  - [`FOMOLocalizationLoss`](#fomolocalizationloss)
- [Instance Segmentation Losses](#instance-segmentation-losses)
  - [`PrecisionDFLSegmentationLoss`](#precisiondflsegmentationloss)
- [Anomaly Detection Losses](#anomaly-detection-losses)
  - [`ReconstructionSegmentationLoss`](#reconstructionsegmentationloss)
- [OCR Losses](#ocr-losses)
  - [`CTCLoss`](#ctcloss)
- [Embedding Losses](#embedding-losses)

## Classification & Segmentation Losses

These losses are applicable to both **segmentation** and **classification** tasks.

### `CrossEntropyLoss`

Adapted from [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

**Parameters:**

| Key               | Type                             | Default value | Description                                                                                                                                                                                                           |
| ----------------- | -------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `weight`          | `list[float] \| None`            | `None`        | Manual rescaling weight for each class. Must be a list of the same length as the number of classes if provided.                                                                                                       |
| `reduction`       | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output.                                                                                                                                                                       |
| `label_smoothing` | `float` (0.0 to 1.0)             | `0.0`         | Smoothing factor applied to targets; creates a mix between the ground truth and a uniform distribution as described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). |

### `BCEWithLogitsLoss`

Adapted from [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

**Parameters:**

| Key          | Type                             | Default value | Description                                                                                                     |
| ------------ | -------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------- |
| `weight`     | `list[float] \| None`            | `None`        | Manual rescaling weight for each class; must be a list of the same length as the number of classes if provided. |
| `reduction`  | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output.                                                                 |
| `pos_weight` | `Tensor \| None`                 | `None`        | Weight for positive examples to be broadcast with the target.                                                   |

### `SmoothBCEWithLogitsLoss`

**Parameters:**

| Key               | Type                             | Default value | Description                                                                                                                                                                        |
| ----------------- | -------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `weight`          | `list[float] \| None`            | `None`        | Manual rescaling weight for each class.                                                                                                                                            |
| `reduction`       | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction to apply to the output.                                                                                                                                    |
| `label_smoothing` | `float` (0.0 to 1.0)             | `0.0`         | Amount of label smoothing; mixes the ground truth with a uniform distribution as in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). |
| `bce_pow`         | `float`                          | `1.0`         | Weight factor for positive samples.                                                                                                                                                |

### `SigmoidFocalLoss`

Adapted from [TorchVision docs](https://pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html).

**Parameters:**

| Key         | Type                             | Default value | Description                                                                         |
| ----------- | -------------------------------- | ------------- | ----------------------------------------------------------------------------------- |
| `alpha`     | `float`                          | `0.25`        | Balancing factor (in the range (0, 1)) to address class imbalance, or -1 to ignore. |
| `gamma`     | `float`                          | `2.0`         | Exponent for the modulating factor `(1 - p_t)` balancing easy versus hard examples. |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction method for the output.                                      |

### `SoftmaxFocalLoss`

**Parameters:**

| Key         | Type                             | Default value | Description                                                                              |
| ----------- | -------------------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| `alpha`     | `float \| list[float]`           | `1`           | Balancing factor; either a single value or a list per class to mitigate class imbalance. |
| `gamma`     | `float`                          | `2.0`         | Exponent for the modulating factor `(1 - p_t)` balancing easy vs. hard examples.         |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction method for the output.                                           |
| `smooth`    | `float`                          | `1e-5`        | Smoothing constant added to labels to avoid zero probabilities.                          |

### `OHEMCrossEntropyLoss`

Wraps the standard `CrossEntropyLoss` with Online Hard Example Mining (OHEM).

**Parameters:**

| Key               | Type                             | Default value | Description                                                                                                                                |
| ----------------- | -------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `ohem_ratio`      | `float`                          | `0.1`         | Fraction of elements (e.g., pixels) to keep based on highest loss values.                                                                  |
| `ohem_threshold`  | `float`                          | `0.7`         | Threshold for hard example selection, computed as `-torch.log(torch.tensor(ohem_threshold))`; only losses above this value are considered. |
| `weight`          | `list[float] \| None`            | `None`        | (Forwarded to `CrossEntropyLoss`) Manual rescaling weight for each class.                                                                  |
| `reduction`       | `Literal["none", "mean", "sum"]` | `"mean"`      | (Forwarded to `CrossEntropyLoss`) Specifies the reduction method.                                                                          |
| `label_smoothing` | `float` (0.0 to 1.0)             | `0.0`         | (Forwarded to `CrossEntropyLoss`) Amount of label smoothing to apply.                                                                      |

### `OHEMBCEWithLogitsLoss`

Wraps the standard `BCEWithLogitsLoss` with Online Hard Example Mining (OHEM).

**Parameters:**

| Key              | Type                             | Default value | Description                                                                                                                                  |
| ---------------- | -------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `ohem_ratio`     | `float`                          | `0.1`         | Fraction of elements to keep based on highest loss values.                                                                                   |
| `ohem_threshold` | `float`                          | `0.7`         | Threshold for hard example selection, computed as `-torch.log(torch.tensor(ohem_threshold))`; only losses above this threshold are retained. |
| `weight`         | `list[float] \| None`            | `None`        | (Forwarded to `BCEWithLogitsLoss`) Manual rescaling weight for each class.                                                                   |
| `reduction`      | `Literal["none", "mean", "sum"]` | `"mean"`      | (Forwarded to `BCEWithLogitsLoss`) Specifies the reduction method.                                                                           |
| `pos_weight`     | `Tensor \| None`                 | `None`        | (Forwarded to `BCEWithLogitsLoss`) Weight for positive examples.                                                                             |

### Bounding Box Detection Losses

#### `AdaptiveDetectionLoss`

Adapted from [this paper](https://arxiv.org/pdf/2209.02976.pdf).

Compatible with: [`EfficientBBoxHead`](../../nodes/README.md#efficientbboxhead)

**Parameters:**

| Key                 | Type                                              | Default value | Description                                                                                                                                                                                                                                              |
| ------------------- | ------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_warmup_epochs`   | `int`                                             | `0`           | Number of epochs using the ATSS assigner before switching to the TAL assigner.                                                                                                                                                                           |
| `iou_type`          | `Literal["none", "giou", "diou", "ciou", "siou"]` | `"giou"`      | Type of IoU used for bounding box regression loss.                                                                                                                                                                                                       |
| `class_loss_weight` | `float`                                           | `1.0`         | Weight for the classification component of the loss.                                                                                                                                                                                                     |
| `iou_loss_weight`   | `float`                                           | `2.5`         | Weight for the IoU regression component of the loss.                                                                                                                                                                                                     |
| `per_class_weights` | `list`                                            | `None`        | A list of weights to scale the loss for each class during training. This allows you to emphasize or de-emphasize certain classes based on their importance or representation in the dataset. The weights' length must be equal to the number of classes. |

#### `PrecisionDFLDetectionLoss`

Adapted from [this paper](https://arxiv.org/pdf/2207.02696.pdf) and [this paper](https://arxiv.org/pdf/2209.02976.pdf).

Compatible with: [`PrecisionBBoxHead`](../../nodes/README.md#precisionbboxhead)

**Parameters:**

| Key                 | Type    | Default value | Description                                    |
| ------------------- | ------- | ------------- | ---------------------------------------------- |
| `tal_topk`          | `int`   | `10`          | Number of anchors considered during selection. |
| `class_loss_weight` | `float` | `0.5`         | Weight for the classification loss.            |
| `bbox_loss_weight`  | `float` | `7.5`         | Weight for the bounding box regression loss.   |
| `dfl_loss_weigth`   | `float` | `1.5`         | Weight for the Distribution Focal Loss (DFL).  |

### Instance Keypoint Detection Losses

#### `EfficientKeypointBBoxLoss`

Adapted from [YOLO-Pose](https://arxiv.org/ftp/arxiv/papers/2204/2204.06806.pdf).

Compatible with: [`EfficientKeypointBBoxHead`](../../nodes/README.md#efficientkeypointbboxhead)

**Parameters:**

| Key                     | Type                                              | Default value | Description                                                                                    |
| ----------------------- | ------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| `iou_type`              | `Literal["none", "giou", "diou", "ciou", "siou"]` | `"giou"`      | IoU type used for the bounding box regression sub-loss.                                        |
| `reduction`             | `Literal["mean", "sum"]`                          | `"mean"`      | Specifies the reduction method for the loss output.                                            |
| `class_loss_weight`     | `float`                                           | `1.0`         | Weight for the classification sub-loss.                                                        |
| `iou_loss_weight`       | `float`                                           | `2.5`         | Weight for the IoU sub-loss.                                                                   |
| `regr_kpts_loss_weight` | `float`                                           | `1.5`         | Weight for the keypoint regression (OKS) sub-loss.                                             |
| `vis_kpts_loss_weight`  | `float`                                           | `2.0`         | Weight for the keypoint visibility sub-loss.                                                   |
| `sigmas`                | `list[float] \| None`                             | `None`        | Sigmas used in KeypointLoss for OKS; if `None`, defaults (e.g., COCO values) are used.         |
| `area_factor`           | `float \| None`                                   | `None`        | Factor to multiply the bounding box area in KeypointLoss; if `None`, a default factor is used. |
| `n_warmup_epochs`       | `int`                                             | `0`           | Number of epochs using the ATSS assigner before switching to the TAL assigner.                 |

#### `FOMOLocalizationLoss`

Compatible with: [`FOMOHead`](../../nodes/README.md#fomohead)

**Parameters:**

| Key             | Type    | Default value | Description                                                                                                                   |
| --------------- | ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `object_weight` | `float` | `500`         | Weight for objects in the loss computation. Higher values may improve detection accuracy at the risk of more false positives. |

### Instance Segmentation Losses

#### `PrecisionDFLSegmentationLoss`

Adapted from [this paper](https://arxiv.org/pdf/2207.02696.pdf) and [this paper](https://arxiv.org/pdf/2209.02976.pdf).

Compatible with: [`PrecisionSegmentBBoxHead`](../../nodes/README.md#precisionsegmentbboxhead)

**Parameters:**

| Key                 | Type    | Default value | Description                                        |
| ------------------- | ------- | ------------- | -------------------------------------------------- |
| `tal_topk`          | `int`   | `10`          | Number of anchors considered during selection.     |
| `class_loss_weight` | `float` | `0.5`         | Weight for the classification loss.                |
| `bbox_loss_weight`  | `float` | `7.5`         | Weight for the bounding box and segmentation loss. |
| `dfl_loss_weigth`   | `float` | `1.5`         | Weight for the Distribution Focal Loss (DFL).      |

## Anomaly Detection Losses

These losses are suited for tasks such as anomaly detection where reconstruction quality is key.

### `ReconstructionSegmentationLoss`

Adapted from [this paper](https://arxiv.org/abs/2108.07610).

Compatible with: [`DiscSubNetHead`](../../nodes/README.md#discsubnethead)

**Parameters:**

| Key         | Type                             | Default value | Description                                                                 |
| ----------- | -------------------------------- | ------------- | --------------------------------------------------------------------------- |
| `alpha`     | `float \| list`                  | `1.0`         | Weighting factor for Focal loss; can be a single value or a list per class. |
| `gamma`     | `float`                          | `2.0`         | Modulates the balance between easy and hard examples in Focal loss.         |
| `reduction` | `Literal["none", "mean", "sum"]` | `"mean"`      | Specifies the reduction method for the output.                              |
| `smooth`    | `float`                          | `1e-5`        | Smoothing factor to prevent overconfidence in predictions.                  |

## OCR Losses

Tailored for Optical Character Recognition tasks.

### `CTCLoss`

CTC loss with optional focal loss weighting.

Compatible with: [`OCRCTCHead`](../../nodes/README.md#ocrctchead)

**Parameters:**

| Key              | Type   | Default value | Description                                            |
| ---------------- | ------ | ------------- | ------------------------------------------------------ |
| `use_focal_loss` | `bool` | `True`        | Whether to apply focal loss weighting to the CTC loss. |

## Embedding Losses

Embedding losses are used in tasks such as metric learning and face recognition, primarily sourced from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/).

Compatible with: [`GhostFaceNetHead`](../../nodes/README.md#ghostfacenethead)

### Available Embedding Losses

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

In addition, the following parameters can be used with any embedding loss:

| Key                  | Type   | Default value | Description                                                                                                                                                                              |
| -------------------- | ------ | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `miner`              | `str`  | `None`        | Name of the miner to use with the loss. All miners from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/miners/) are supported.                        |
| `miner_params`       | `dict` | `None`        | Parameters for the miner.                                                                                                                                                                |
| `distance`           | `str`  | `None`        | Name of the distance metric to use with the loss. All distance metrics from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/distances/) are supported. |
| `distance_params`    | `dict` | `None`        | Parameters for the distance metric.                                                                                                                                                      |
| `reducer`            | `str`  | `None`        | Name of the reducer to use with the loss. All reducers from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/reducers/) are supported.                  |
| `reducer_params`     | `dict` | `None`        | Parameters for the reducer.                                                                                                                                                              |
| `regularizer`        | `str`  | `None`        | Name of the regularizer to use with the loss. All regularizers from [pytorch-metric-learning](https://kevinmusgrave.github.io/pytorch-metric-learning/regularizers/) are supported.      |
| `regularizer_params` | `dict` | `None`        | Parameters for the regularizer.                                                                                                                                                          |
