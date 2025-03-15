# Metrics

List of all the available metrics.

## Table Of Contents

- [Accuracy](#accuracy)
- [JaccardIndex](#jaccardindex)
- [F1Score](#f1score)
- [Precision](#precision)
- [Recall](#recall)
- [ObjectKeypointSimilarity](#objectkeypointsimilarity)
- [MeanAveragePrecision](#meanaverageprecision)
- [ClosestIsPositiveAccuracy](#closestispositiveaccuracy)
- [MedianDistances](#mediandistances)
- [OCRAccuracy](#ocraccuracy)
- [ConfusionMatrix](#confusionmatrix)

## Accuracy

Accuracy metric from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html) module.

Works with tasks such as **classification, segmentation, and anomaly detection.**

## JaccardIndex

Jaccard Index (Intersection over Union) metric from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html) module.

Works with tasks such as **classification, segmentation, and anomaly detection.**

## F1Score

F1 Score metric from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html) module.

Works with tasks such as **classification, segmentation, and anomaly detection.**

## Precision

Precision metric from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html) module.

Works with tasks such as **classification, segmentation, and anomaly detection.**

## Recall

Recall metric from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html) module.

Works with tasks such as **classification, segmentation, and anomaly detection.**

> **Note:** For multi-label classification, ensure that you specify the `params.task` as `multilabel` when using these metrics.

## ObjectKeypointSimilarity

For more information, see [object-keypoint-similarity](https://learnopencv.com/object-keypoint-similarity/).

Works with **keypoint detection task.**

**Params**

| Key                | Type                  | Default value | Description                                                                                                                                                         |
| ------------------ | --------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sigmas`           | `list[float] \| None` | `None`        | List of sigma values for each keypoint. If `None`, the COCO sigmas are used when the COCO dataset is provided. Otherwise, a default sigma value of 0.04 is applied. |
| `area_factor`      | `float`               | `0.53`        | Factor by which to multiply the bounding box area                                                                                                                   |
| `use_cocoeval_oks` | `bool`                | `True`        | Whether to use the same OKS formula as in COCO evaluation                                                                                                           |

> \[!NOTE\]
> **Important:** The ObjectKeypointSimilarity metric is sensitive to NMS parameters, such as confidence and IoU thresholds, as well as to sigmas that are also set in the loss computation and represent the uncertainty in keypoints. Make sure to adjust these settings appropriately for your specific use case.

## MeanAveragePrecision

Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` for object detection tasks. This metric is built upon [pycocotools](https://github.com/cocodataset/cocoapi), which provides robust tools for evaluating these tasks.

Is compatible with **object and keypoint detection** and **instance segmentation tasks.**

```math
\text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i
```

where $AP_i$ is the average precision for class $i$ and $n$ is the number of classes. Average precision is the area under the precision-recall curve. For both detection and segmentation tasks, precision and recall are computed based on the Intersection over Union (IoU) between predicted and ground truth regions (bounding boxes or masks). A prediction is considered a true positive if its IoU exceeds a threshold $t$. Precision is defined as the ratio of true positives to all detections, and recall as the ratio of true positives to all ground truth instances.

> \[!NOTE\]
> **Important:** Mean Average Precision metric is sensitive to NMS parameters, such as confidence and IoU thresholds. Make sure to adjust these settings appropriately for your specific use case.

**Detection and Instance Segmentation Params**
See [Mean Average Precision](https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html) for more information.

**Instance Keypoint Params**

| Key           | Type                                | Default value | Description                                                           |
| ------------- | ----------------------------------- | ------------- | --------------------------------------------------------------------- |
| `sigmas`      | `list[float] \| None`               | `None`        | List of sigmas for each keypoint. If `None`, the COCO sigmas are used |
| `area_factor` | `float`                             | `0.53`        | Factor by which to multiply the bounding box area                     |
| `max_dets`    | `int`                               | `20`          | Maximum number of detections per image                                |
| `box_fotmat`  | `Literal["xyxy", "xywh", "cxcywh"]` | `"xyxy"`      | Format of the bounding boxes                                          |

> \[!NOTE\]
> **Important:** Mean Average Precision Keypoints metric is sensitive to NMS parameters, such as confidence and IoU thresholds. Make sure to adjust these settings appropriately for your specific use case.

## ClosestIsPositiveAccuracy

Compute the accuracy of the closest positive sample to the query sample.
Needs to be connected to the `GhostFaceNetHead` node.

Works with **embedding task**.

## MedianDistances

Compute the median distance between the query and the positive samples.
Needs to be connected to the `GhostFaceNetHead` node.

Works with **embedding task**.

## OCRAccuracy

Works with **OCR tasks**.

**Params**

| Key         | Type  | Default value | Description                                |
| ----------- | ----- | ------------- | ------------------------------------------ |
| `blank_cls` | `int` | `0`           | Index of the blank class. Defaults to `0`. |

## ConfusionMatrix

Works with **classification, segmentation, object detection, instance keypoint detection and instance segmentation tasks**

> \[!NOTE\]
> **Important:** Confusion Matrix is sensitive to NMS parameters, such as confidence and IoU thresholds. Make sure to adjust these settings appropriately for your specific use case.
>
> **Note:** The Confusion Matrix should not be used as the primary metric for model evaluation.
