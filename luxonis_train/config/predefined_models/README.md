# Predefined models

In addition to defining the model by hand, we offer a list of simple predefined
models which can be used instead.

## Table Of Contents

- [`SegmentationModel`](#segmentationmodel)
- [`DetectionModel`](#detectionmodel)
- [`KeypointDetectionModel`](#keypointdetectionmodel)
- [`ClassificationModel`](#classificationmodel)

**Parameters:**

| Key                   | Type             | Default value | Description                                                          |
| --------------------- | ---------------- | ------------- | -------------------------------------------------------------------- |
| `name`                | `str`            | -             | Name of the predefined architecture. See below the available options |
| `params`              | `dict[str, Any]` | `{}`          | Additional parameters of the predefined model                        |
| `include_nodes`       | `bool`           | `True`        | Whether to include nodes of the model                                |
| `include_losses`      | `bool`           | `True`        | Whether to include loss functions                                    |
| `include_metrics`     | `bool`           | `True`        | Whether to include metrics                                           |
| `include_visualizers` | `bool`           | `True`        | Whether to include visualizers                                       |

## `SegmentationModel`

The `SegmentationModel` allows for both `"light"` and `"heavy"` variants, where the `"heavy"` variant is more accurate, and the `"light"` variant is faster.

See an example configuration file using this predefined model [here](../../../configs/segmentation_light_model.yaml) for the `"light"` variant, and [here](../../../configs/segmentation_heavy_model.yaml) for the `"heavy"` variant.

**Components:**

| Name                                                                                            | Alias                          | Function                                                                                            |
| ----------------------------------------------------------------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------- |
| [`DDRNet`](../../nodes/README.md#ddrnet)                                                        | `"segmentation_backbone"`      | Backbone of the model. Available variants: `"light"` (`DDRNet-23-slim`) and `"heavy"` (`DDRNet-23`) |
| [`SegmentationHead`](../../nodes/README.md#segmentationhead)                                    | `"segmentation_head"`          | Head of the model                                                                                   |
| [`BCEWithLogitsLoss`](../../attached_modules/losses/README.md#bcewithlogitsloss)                | `"segmentation_loss"`          | Loss of the model when the task is set to `"binary"`                                                |
| [`CrossEntropyLoss`](../../attached_modules/losses/README.md#crossentropyloss)                  | `"segmentation_loss"`          | Loss of the model when the task is set to `"multiclass"` or `"multilabel"`                          |
| [`JaccardIndex`](../../attached_modules/metrics/README.md#torchmetrics)                         | `"segmentation_jaccard_index"` | Main metric of the model                                                                            |
| [`F1Score`](../../attached_modules/metrics/README.md#torchmetrics)                              | `"segmentation_f1_score"`      | Secondary metric of the model                                                                       |
| [`SegmentationVisualizer`](../../attached_modules/visualizers/README.md#segmentationvisualizer) | `"segmentation_visualizer"`    | Visualizer of the `SegmentationHead`                                                                |

**Parameters:**

| Key                 | Type                              | Default value | Description                                                                                     |
| ------------------- | --------------------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `variant`           | `Literal["light", "heavy"]`       | `"light"`     | Defines the variant of the model. `"light"` uses `DDRNet-23-slim`, `"heavy"` uses `DDRNet-23`   |
| `backbone`          | `str`                             | `"DDRNet"`    | Name of the node to be used as a backbone                                                       |
| `backbone_params`   | `dict`                            | `{}`          | Additional parameters for the backbone. If not provided, variant-specific defaults will be used |
| `head_params`       | `dict`                            | `{}`          | Additional parameters for the head                                                              |
| `aux_head_params`   | `dict`                            | `{}`          | Additional parameters for auxiliary heads                                                       |
| `loss_params`       | `dict`                            | `{}`          | Additional parameters for the loss                                                              |
| `visualizer_params` | `dict`                            | `{}`          | Additional parameters for the visualizer                                                        |
| `task`              | `Literal["binary", "multiclass"]` | `"binary"`    | Type of the task of the model                                                                   |
| `task_name`         | `str \| None`                     | `None`        | Custom task name for the head                                                                   |

## `DetectionModel`

The `DetectionModel` allows for both `"light"` and `"heavy"` variants, where the `"heavy"` variant is more accurate, and the `"light"` variant is faster.

See an example configuration file using this predefined model [here](../../../configs/detection_light_model.yaml) for the `"light"` variant, and [here](../../../configs/detection_heavy_model.yaml) for the `"heavy"` variant.

**Components:**

| Name                                                                                     | Alias                    | Function                                                                                                 |
| ---------------------------------------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------------------------------------- |
| [`EfficientRep`](../../nodes/README.md#efficientrep)                                     | `"detection_backbone"`   | Backbone of the model. Available variants: `"light"` (`EfficientRep-N`) and `"heavy"` (`EfficientRep-L`) |
| [`RepPANNeck`](../../nodes/README.md#reppanneck)                                         | `"detection_neck"`       | Neck of the model                                                                                        |
| [`EfficientBBoxHead`](../../nodes/README.md#efficientbboxhead)                           | `"detection_head"`       | Head of the model                                                                                        |
| [`AdaptiveDetectionLoss`](../../attached_modules/losses/README.md#adaptivedetectionloss) | `"detection_loss"`       | Loss of the model                                                                                        |
| [`MeanAveragePrecision`](../../attached_modules/metrics/README.md#meanaverageprecision)  | `"detection_map"`        | Main metric of the model                                                                                 |
| [`BBoxVisualizer`](../../attached_modules/visualizers/README.md#bboxvisualizer)          | `"detection_visualizer"` | Visualizer of the `detection_head`                                                                       |

**Parameters:**

| Key                 | Type                        | Default value    | Description                                                                                        |
| ------------------- | --------------------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `variant`           | `Literal["light", "heavy"]` | `"light"`        | Defines the variant of the model. `"light"` uses `EfficientRep-N`, `"heavy"` uses `EfficientRep-L` |
| `use_neck`          | `bool`                      | `True`           | Whether to include the neck in the model                                                           |
| `backbone`          | `str`                       | `"EfficientRep"` | Name of the node to be used as a backbone                                                          |
| `backbone_params`   | `dict`                      | `{}`             | Additional parameters to the backbone                                                              |
| `neck_params`       | `dict`                      | `{}`             | Additional parameters to the neck                                                                  |
| `head_params`       | `dict`                      | `{}`             | Additional parameters to the head                                                                  |
| `loss_params`       | `dict`                      | `{}`             | Additional parameters to the loss                                                                  |
| `visualizer_params` | `dict`                      | `{}`             | Additional parameters to the visualizer                                                            |
| `task_name`         | `str \| None`               | `None`           | Custom task name for the head                                                                      |

## `KeypointDetectionModel`

The `KeypointDetectionModel` allows for both `"light"` and `"heavy"` variants, where the `"heavy"` variant is more accurate, and the `"light"` variant is faster.

See an example configuration file using this predefined model [here](../../../configs/keypoint_bbox_light_model.yaml) for the `"light"` variant, and [here](../../../configs/keypoint_bbox_heavy_model.yaml) for the `"heavy"` variant.

**Components:**

| Name                                                                                                      | Alias                        | Function                                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`EfficientRep`](../../nodes/README.md#efficientrep)                                                      | `"kpt_detection_backbone"`   | Backbone of the model. Available variants: `"light"` (`EfficientRep-N`) and `"heavy"` (`EfficientRep-L`)                                                 |
| [`RepPANNeck`](../../nodes/README.md#reppanneck)                                                          | `"kpt_detection_neck"`       | Neck of the model                                                                                                                                        |
| [`EfficientKeypointBBoxHead`](../../nodes/README.md#efficientkeypointbboxhead)                            | `"kpt_detection_head"`       | Head of the model                                                                                                                                        |
| [`EfficientKeypointBBoxLoss`](../../attached_modules/losses/README.md#efficientkeypointbboxloss)          | `"kpt_detection_loss"`       | Loss of the model                                                                                                                                        |
| [`ObjectKeypointSimilarity`](../../attached_modules/metrics/README.md#objectkeypointsimilarity)           | `"kpt_detection_oks"`        | Main metric of the model                                                                                                                                 |
| [`MeanAveragePrecisionKeypoints`](../../attached_modules/metrics/README.md#meanaverageprecisionkeypoints) | `"kpt_detection_map"`        | Secondary metric of the model                                                                                                                            |
| [`BBoxVisualizer`](../../attached_modules/visualizers/README.md#bboxvisualizer)                           | `"kpt_detection_visualizer"` | Visualizer for bounding boxes. Combined with keypoint visualizer using [`MultiVisualizer`](../../attached_modules/visualizers/README.md#multivisualizer) |
| [`KeypointVisualizer`](../../attached_modules/visualizers/README.md#keypointvisualizer)                   | `"kpt_detection_visualizer"` | Visualizer for keypoints. Combined with keypoint visualizer using [`MultiVisualizer`](../../attached_modules/visualizers/README.md#multivisualizer)      |

**Parameters:**

| Key                      | Type                        | Default value    | Description                                                                                        |
| ------------------------ | --------------------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `variant`                | `Literal["light", "heavy"]` | `"light"`        | Defines the variant of the model. `"light"` uses `EfficientRep-N`, `"heavy"` uses `EfficientRep-L` |
| `use_neck`               | `bool`                      | `True`           | Whether to include the neck in the model                                                           |
| `backbone`               | `str`                       | `"EfficientRep"` | Name of the node to be used as a backbone                                                          |
| `backbone_params`        | `dict`                      | `{}`             | Additional parameters to the backbone                                                              |
| `neck_params`            | `dict`                      | `{}`             | Additional parameters to the neck                                                                  |
| `head_params`            | `dict`                      | `{}`             | Additional parameters to the head                                                                  |
| `loss_params`            | `dict`                      | `{}`             | Additional parameters to the loss                                                                  |
| `kpt_visualizer_params`  | `dict`                      | `{}`             | Additional parameters to the keypoint visualizer                                                   |
| `bbox_visualizer_params` | `dict`                      | `{}`             | Additional parameters to the bounding box visualizer                                               |
| `bbox_task_name`         | `str \| None`               | `None`           | Custom task name for the detection head                                                            |
| `kpt_task_name`          | `str \| None`               | `None`           | Custom task name for the keypoint head                                                             |

## `ClassificationModel`

The `ClassificationModel` allows for both `"light"` and `"heavy"` variants, where the `"heavy"` variant is more accurate, and the `"light"` variant is faster. Can be used for multi-class and multi-label tasks.

See an example configuration file using this predefined model [here](../../../configs/classification_light_model.yaml) for the `"light"` variant, and [here](../../../configs/classification_heavy_model.yaml) for the `"heavy"` variant.

**Components:**

| Name                                                                           | Alias                       | Function                                                                                                     |
| ------------------------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| [`ResNet`](../../nodes/README.md#resnet)                                       | `"classification_backbone"` | Backbone of the model. The `"light"` variant uses `ResNet-18`, while the `"heavy"` variant uses `ResNet-101` |
| [`ClassificationHead`](../../nodes/README.md#classificationhead)               | `"classification_head"`     | Head of the model                                                                                            |
| [`CrossEntropyLoss`](../../attached_modules/losses/README.md#crossentropyloss) | `"classification_loss"`     | Loss of the model                                                                                            |
| [F1Score](../../attached_modules/metrics/README.md#torchmetrics)               | `"classification_f1_score"` | Main metric of the model                                                                                     |
| [Accuracy](../../attached_modules/metrics/README.md#torchmetrics)              | `"classification_accuracy"` | Secondary metric of the model                                                                                |
| [Recall](../../attached_modules/metrics/README.md#torchmetrics)                | `"classification_recall"`   | Secondary metric of the model                                                                                |

**Parameters:**

| Key                 | Type                                  | Default value  | Description                                                                               |
| ------------------- | ------------------------------------- | -------------- | ----------------------------------------------------------------------------------------- |
| `variant`           | `Literal["light", "heavy"]`           | `"light"`      | Defines the variant of the model. `"light"` uses `ResNet-18`, `"heavy"` uses `ResNet-101` |
| `backbone`          | `str`                                 | `"ResNet"`     | Name of the node to be used as a backbone                                                 |
| `backbone_params`   | `dict`                                | `{}`           | Additional parameters to the backbone                                                     |
| `head_params`       | `dict`                                | `{}`           | Additional parameters to the head                                                         |
| `loss_params`       | `dict`                                | `{}`           | Additional parameters to the loss                                                         |
| `visualizer_params` | `dict`                                | `{}`           | Additional parameters to the visualizer                                                   |
| `task`              | `Literal["multiclass", "multilabel"]` | `"multiclass"` | Type of the task of the model                                                             |
| `task_name`         | `str \| None`                         | `None`         | Custom task name for the head                                                             |