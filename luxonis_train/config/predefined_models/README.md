# Predefined models

In addition to defining the model by hand, we offer a list of simple predefined
models which can be used instead.

## Table Of Contents

- [`SegmentationModel`](#segmentationmodel)
- [`DetectionModel`](#detectionmodel)
- [`KeypointDetectionModel`](#keypointdetectionmodel)
- [`ClassificationModel`](#classificationmodel)
- [`FOMOModel`](#fomomodel)
- [`InstanceSegmentationModel`](#instancesegmentationmodel)
- [`AnomalyDetectionModel`](#anomalydetectionmodel)

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

The `SegmentationModel` supports `"light"` and `"heavy"` variants, with `"light"` optimized for speed and `"heavy"` for accuracy.

See an example configuration file using this predefined model [here](../../../configs/segmentation_light_model.yaml) for the `"light"` variant, and [here](../../../configs/segmentation_heavy_model.yaml) for the `"heavy"` variant.

### Performance Metrics

FPS (frames per second) for `light` and `heavy` variants on different devices with image size 384x512:

| Variant     | RVC2 FPS | RVC4 FPS |
| ----------- | -------- | -------- |
| **`light`** | 43       | 75       |
| **`heavy`** | 14       | 48       |

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

The `DetectionModel` supports `"light"`, `"medium"`, and `"heavy"` variants, with `"light"` optimized for speed, `"heavy"` for accuracy, and `"medium"` offering a balance between the two.

See an example configuration file using this predefined model [here](../../../configs/detection_light_model.yaml) for the `"light"` variant, and [here](../../../configs/detection_heavy_model.yaml) for the `"heavy"` variant.

This detection model is based on [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/pdf/2209.02976.pdf).

The pretrained models achieve the following performance on the COCOval2017 dataset:

- **`light` variant**: **36.1% mAP** (trained on COCOtrain2017)
- **`heavy` variant**: **51.2% mAP** (trained on COCOtrain2017)

**Note**: To align with the evaluation procedure used by other state-of-the-art (SOTA) models, we adopted a small `conf_thresh` (e.g., `0.03`) and a high `iou_thresh` (e.g., `0.65`) during validation.

### Performance Metrics

FPS (frames per second) for `light`, `medium` and `heavy` variants on different devices with image size 384x512:

| Variant      | RVC2 FPS | RVC4 FPS |
| ------------ | -------- | -------- |
| **`light`**  | 50       | 194      |
| **`medium`** | 25       | 166      |
| **`heavy`**  | 7        | 120      |

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

| Key                 | Type                                  | Default value    | Description                                                                                        |
| ------------------- | ------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `variant`           | `Literal["light", "heavy", "medium"]` | `"light"`        | Defines the variant of the model. `"light"` uses `EfficientRep-N`, `"heavy"` uses `EfficientRep-L` |
| `use_neck`          | `bool`                                | `True`           | Whether to include the neck in the model                                                           |
| `backbone`          | `str`                                 | `"EfficientRep"` | Name of the node to be used as a backbone                                                          |
| `backbone_params`   | `dict`                                | `{}`             | Additional parameters to the backbone                                                              |
| `neck_params`       | `dict`                                | `{}`             | Additional parameters to the neck                                                                  |
| `head_params`       | `dict`                                | `{}`             | Additional parameters to the head                                                                  |
| `loss_params`       | `dict`                                | `{}`             | Additional parameters to the loss                                                                  |
| `visualizer_params` | `dict`                                | `{}`             | Additional parameters to the visualizer                                                            |
| `task_name`         | `str \| None`                         | `None`           | Custom task name for the head                                                                      |

## `KeypointDetectionModel`

The `KeypointDetectionModel` supports `"light"`, `"medium"`, and `"heavy"` variants, with `"light"` optimized for speed, `"heavy"` for accuracy, and `"medium"` offering a balance between the two.

See an example configuration file using this predefined model [here](../../../configs/keypoint_bbox_light_model.yaml) for the `"light"` variant, and [here](../../../configs/keypoint_bbox_heavy_model.yaml) for the `"heavy"` variant.

### Performance Metrics

FPS (frames per second) for `light`, `medium` and `heavy` variants on different devices with image size 384x512:

| Variant      | RVC2 FPS | RVC4 FPS |
| ------------ | -------- | -------- |
| **`light`**  | 45       | 137      |
| **`medium`** | 24       | 120      |
| **`heavy`**  | 7        | 101      |

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

| Key                      | Type                                  | Default value    | Description                                                                                        |
| ------------------------ | ------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------- |
| `variant`                | `Literal["light", "heavy", "medium"]` | `"light"`        | Defines the variant of the model. `"light"` uses `EfficientRep-N`, `"heavy"` uses `EfficientRep-L` |
| `use_neck`               | `bool`                                | `True`           | Whether to include the neck in the model                                                           |
| `backbone`               | `str`                                 | `"EfficientRep"` | Name of the node to be used as a backbone                                                          |
| `backbone_params`        | `dict`                                | `{}`             | Additional parameters to the backbone                                                              |
| `neck_params`            | `dict`                                | `{}`             | Additional parameters to the neck                                                                  |
| `head_params`            | `dict`                                | `{}`             | Additional parameters to the head                                                                  |
| `loss_params`            | `dict`                                | `{}`             | Additional parameters to the loss                                                                  |
| `kpt_visualizer_params`  | `dict`                                | `{}`             | Additional parameters to the keypoint visualizer                                                   |
| `bbox_visualizer_params` | `dict`                                | `{}`             | Additional parameters to the bounding box visualizer                                               |
| `bbox_task_name`         | `str \| None`                         | `None`           | Custom task name for the detection head                                                            |
| `kpt_task_name`          | `str \| None`                         | `None`           | Custom task name for the keypoint head                                                             |

## `ClassificationModel`

The `ClassificationModel` supports `"light"` and `"heavy"` variants, with `"light"` optimized for speed and `"heavy"` for accuracy.

See an example configuration file using this predefined model [here](../../../configs/classification_light_model.yaml) for the `"light"` variant, and [here](../../../configs/classification_heavy_model.yaml) for the `"heavy"` variant.

### Performance Metrics

FPS (frames per second) for `light` and `heavy` variants on different devices with image size 384x512:

| Variant     | RVC2 FPS | RVC4 FPS |
| ----------- | -------- | -------- |
| **`light`** | 46       | 176      |
| **`heavy`** | 5        | 134      |

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

## `FOMOModel`

The `FOMOModel` supports `"light"` and `"heavy"` variants, with `"light"` optimized for speed and `"heavy"` for accuracy.

There is a trade-off in this simple model: training with a larger `object_weight` in the loss parameters may result in more false positives (FP), but it will improve accuracy. You can also use `use_nms: True` in the `head_params` to enable NMS which can reduce FP, but it will also reduce TP for close neighbors.

### Performance Metrics

FPS (frames per second) for `light` and `heavy` variants on different devices with image size 384x512:

| Variant     | RVC2 FPS | RVC4 FPS |
| ----------- | -------- | -------- |
| **`light`** | 140      | 243      |
| **`heavy`** | 34       | 230      |

There is a trade-off in this simple model: training with a larger `object_weight` in the loss parameters may result in more false positives (FP), but it will improve accuracy. You can also use `use_nms: True` in the `head_params` to enable NMS which can reduce FP, but it will also reduce TP for close neighbors.

For larger heatmaps and improved accuracy, you can adjust the `attach_index` in the `head_params` to a lower value. This will connect the head to an earlier layer in the backbone, resulting in larger heatmaps. However, be aware that this may lead to slower inference times.

### **Components**

| Name                                                                                            | Alias               | Function                                                                       |
| ----------------------------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------ |
| [`MobileNetV2`](../../nodes/README.md#mobilenetv2)                                              | `"fomo_backbone"`   | Backbone of the model. Available for `"heavy"` variant                         |
| [`EfficientRep`](../../nodes/README.md#efficientrep)                                            | `"fomo_backbone"`   | Backbone of the model. Available for `"light"` variant                         |
| [`FOMOHead`](../../nodes/README.md#fomohead)                                                    | `"fomo_head"`       | Head of the model with a configurable number of convolutional layers.          |
| [`FOMOLocalizationLoss`](../../attached_modules/losses/README.md#fomolocalizationloss)          | `"fomo_loss"`       | Loss function for object localization.                                         |
| [`ObjectKeypointSimilarity`](../../attached_modules/metrics/README.md#objectkeypointsimilarity) | `"fomo_oks"`        | Metric to evaluate the similarity between predicted and true object keypoints. |
| [`KeypointVisualizer`](../../attached_modules/visualizers/README.md#keypointvisualizer)         | `"fomo_visualizer"` | Visualizer for the model's keypoint predictions.                               |

### **Parameters**

| Key                 | Type                        | Default value   | Description                                                                                     |
| ------------------- | --------------------------- | --------------- | ----------------------------------------------------------------------------------------------- |
| `variant`           | `Literal["light", "heavy"]` | `"light"`       | Defines the variant of the model. `"light"` uses fewer layers in the head, `"heavy"` uses more. |
| `backbone`          | `str`                       | `"MobileNetV2"` | Name of the node to be used as a backbone.                                                      |
| `backbone_params`   | `dict`                      | `{}`            | Additional parameters for the backbone.                                                         |
| `head_params`       | `dict`                      | `{}`            | Additional parameters for the head, such as the number of convolutional layers.                 |
| `loss_params`       | `dict`                      | `{}`            | Additional parameters for the loss function.                                                    |
| `visualizer_params` | `dict`                      | `{}`            | Additional parameters for the visualizer.                                                       |
| `task_name`         | `str \| None`               | `None`          | Custom task name for the model head.                                                            |

## `InstanceSegmentationModel`

The `InstanceSegmentationModel` supports `"light"`, `"medium"`, and `"heavy"` variants, with `"light"` optimized for speed, `"heavy"` for accuracy, and `"medium"` offering a balance between the two.

See an example configuration file using this predefined model [here](../../../configs/instance_segmentation_light_model.yaml) for the `"light"` variant, and [here](../../../configs/instance_segmentation_heavy_model.yaml) for the `"heavy"` variant.

### Performance Metrics

FPS (frames per second) for `light`, `medium` and `heavy` variants on different devices with image size 384x512:

| Variant      | RVC2 FPS | RVC4 FPS |
| ------------ | -------- | -------- |
| **`light`**  | 15       | 131      |
| **`medium`** | 9        | 116      |
| **`heavy`**  | 3        | 82       |

**Components:**

| Name                                                                                                            | Alias                                | Function                                                                                                                                 |
| --------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [`EfficientRep`](../../nodes/README.md#efficientrep)                                                            | `"instance_segmentation_backbone"`   | Backbone of the model. Available variants: `"light"` (`EfficientRep-N`), `"medium"` (`EfficientRep-S`), and `"heavy"` (`EfficientRep-L`) |
| [`RepPANNeck`](../../nodes/README.md#reppanneck)                                                                | `"instance_segmentation_neck"`       | Neck of the model                                                                                                                        |
| [`PrecisionSegmentBBoxHead`](../../nodes/README.md#precisionsegmentbboxhead)                                    | `"instance_segmentation_head"`       | Head of the model for instance segmentation                                                                                              |
| [`PrecisionDFLSegmentationLoss`](../../attached_modules/losses/README.md#precisiondflsegmentationloss)          | `"instance_segmentation_loss"`       | Loss function for training instance segmentation models                                                                                  |
| [`MeanAveragePrecision`](../../attached_modules/metrics/README.md#meanaverageprecision)                         | `"instance_segmentation_map"`        | Main metric of the model, measuring mean average precision                                                                               |
| [`InstanceSegmentationVisualizer`](../../attached_modules/visualizers/README.md#instancesegmentationvisualizer) | `"instance_segmentation_visualizer"` | Visualizer for displaying instance segmentation results                                                                                  |

**Parameters:**

| Key                 | Type                                  | Default value    | Description                                                                                                                          |
| ------------------- | ------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `variant`           | `Literal["light", "medium", "heavy"]` | `"light"`        | Defines the variant of the model. `"light"` uses `EfficientRep-N`, `"medium"` uses `EfficientRep-S`, `"heavy"` uses `EfficientRep-L` |
| `use_neck`          | `bool`                                | `True`           | Whether to include the neck in the model                                                                                             |
| `backbone`          | `str`                                 | `"EfficientRep"` | Name of the node to be used as a backbone                                                                                            |
| `backbone_params`   | `dict`                                | `{}`             | Additional parameters to the backbone                                                                                                |
| `neck_params`       | `dict`                                | `{}`             | Additional parameters to the neck                                                                                                    |
| `head_params`       | `dict`                                | `{}`             | Additional parameters to the head                                                                                                    |
| `loss_params`       | `dict`                                | `{}`             | Additional parameters to the loss function                                                                                           |
| `visualizer_params` | `dict`                                | `{}`             | Additional parameters to the visualizer                                                                                              |
| `task_name`         | `str \| None`                         | `None`           | Custom task name for the head                                                                                                        |

## `AnomalyDetectionModel`

The `AnomalyDetectionModel` supports `"light"` and `"heavy"` variants, with `"light"` optimized for speed and `"heavy"` for accuracy.

### Performance Metrics

FPS (frames per second) for `light` and `heavy` variants on different devices with image size 256x256:

| Variant     | RVC2 FPS | RVC4 FPS |
| ----------- | -------- | -------- |
| **`light`** | 3        | 120      |
| **`heavy`** | 0.5      | 61       |

**Components:**

| Name                                                                                                       | Alias                     | Function                                                           |
| ---------------------------------------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| [`RecSubNet`](../../nodes/README.md#recsubnet)                                                             | `"anomaly_backbone"`      | Backbone of the model. Available variants: `"light"` and `"heavy"` |
| [`DiscSubNetHead`](../../nodes/README.md#discsubnethead)                                                   | `"anomaly_head"`          | Head of the model                                                  |
| [`ReconstructionSegmentationLoss`](../../attached_modules/losses/README.md#reconstructionsegmentationloss) | `"anomaly_loss"`          | Loss of the model                                                  |
| [`JaccardIndex`](../../attached_modules/metrics/README.md#torchmetrics)                                    | `"anomaly_jaccard_index"` | Main metric of the model                                           |
| [`SegmentationVisualizer`](../../attached_modules/visualizers/README.md#segmentationvisualizer)            | `"anomaly_visualizer"`    | Visualizer of the `DiscSubNetHead`                                 |

**Parameters:**

| Key                  | Type                        | Default value | Description                                                                                     |
| -------------------- | --------------------------- | ------------- | ----------------------------------------------------------------------------------------------- |
| `variant`            | `Literal["light", "heavy"]` | `"light"`     | Defines the variant of the model. `"light"` uses a smaller network, `"heavy"` uses a larger one |
| `backbone`           | `str`                       | `"RecSubNet"` | Name of the node to be used as a backbone                                                       |
| `backbone_params`    | `dict`                      | `{}`          | Additional parameters for the backbone                                                          |
| `disc_subnet_params` | `dict`                      | `{}`          | Additional parameters for the discriminator subnet                                              |
| `loss_params`        | `dict`                      | `{}`          | Additional parameters for the loss                                                              |
| `visualizer_params`  | `dict`                      | `{}`          | Additional parameters for the visualizer                                                        |
| `task_name`          | `str \| None`               | `None`        | Custom task name for the head                                                                   |

## `OCRRecognitionModel`

This model is based on the [PPOCRv4](https://github.com/PaddlePaddle/PaddleOCR) recognition model. In order to create a dataset you need to provide image paths, labels and text lengths. Each label should be a string with the text that is present in the image. The text length should be the length of the text in the image. You can use the following code snippet to create a dataset:

```python
for path, label in tqdm(zip(im_paths, labels)):
    if len(label):
        yield {
            "file": path,
            "annotation": {
                "metadata": {"text": label},
            },
        }
```

### Performance Metrics

FPS for `light` variant on different devices with image size 48x320:

| Variant     | RVC2 FPS | RVC4 FPS |
| ----------- | -------- | -------- |
| **`light`** | 77       | 350      |

**Components:**

| Name                                                                          | Alias                             | Function                                                             |
| ----------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| [`PPLCNetV3`](../../nodes/README.md#pplcnetv3)                                | `"ocr_recognition"`               | Backbone of the OCR recognition model.                               |
| [`SVTRNeck`](../../nodes/README.md#svtrneck)                                  | `"ocr_recognition/SVTRNeck"`      | Neck component to refine features from the backbone.                 |
| [`OCRCTCHead`](../../nodes/README.md#ocrctchead)                              | `"ocr_recognition/OCRCTCHead"`    | Head of the model for text recognition with CTC decoding.            |
| [`CTCLoss`](../../attached_modules/losses/README.md#ctcloss)                  | `"ocr_recognition/CTCLoss"`       | Loss function for sequence learning in OCR recognition tasks.        |
| [`OCRAccuracy`](../../attached_modules/metrics/README.md#ocraccuracy)         | `"ocr_recognition/OCRAccuracy"`   | Metric to evaluate the accuracy of text recognition.                 |
| [`OCRVisualizer`](../../attached_modules/visualizers/README.md#ocrvisualizer) | `"ocr_recognition/OCRVisualizer"` | Visualizer to display the predicted text alongside the input images. |

**Parameters:**

| Key                 | Type               | Default value | Description                                                 |
| ------------------- | ------------------ | ------------- | ----------------------------------------------------------- |
| `variant`           | `str`              | `"light"`     | Defines the variant of the model.                           |
| `backbone`          | `str`              | `"SVTRNeck"`  | Name of the node to be used as a backbone                   |
| `backbone_params`   | `dict`             | `{}`          | Additional parameters for the backbone                      |
| `neck_params`       | `dict`             | `{}`          | Additional parameters for the neck                          |
| `head_params`       | `dict`             | `{}`          | Additional parameters for the head                          |
| `loss_params`       | `dict`             | `{}`          | Additional parameters for the loss                          |
| `visualizer_params` | `dict`             | `{}`          | Additional parameters for the visualizer                    |
| `task_name`         | `str \| None`      | `None`        | Custom task name for the head                               |
| `alphabet`          | `List[str] \| str` | `english`     | List of characters or a name of a predefined alphabet.      |
| `max_text_len`      | `int`              | `40`          | Maximum text length.                                        |
| `ignore_unknown`    | `bool`             | `True`        | Whether to ignore unknown characters (not in the alphabet). |
