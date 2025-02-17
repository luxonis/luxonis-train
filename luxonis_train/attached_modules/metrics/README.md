# Metrics

List of all the available metrics.

## Table Of Contents

- [Torchmetrics](#torchmetrics)
- [ObjectKeypointSimilarity](#objectkeypointsimilarity)
- [MeanAveragePrecision](#meanaverageprecision)
- [MeanAveragePrecisionKeypoints](#meanaverageprecisionkeypoints)
- [ClosestIsPositiveAccuracy](#closestispositiveaccuracy)
- [MedianDistances](#mediandistances)

## Torchmetrics

Metrics from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/) module.

- [Accuracy](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html)
- [JaccardIndex](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html) -- Intersection over Union.
- [F1Score](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)
- [Precision](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html)
- [Recall](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html)

> **Note:** For multi-label classification, ensure that you specify the `params.task` as `multilabel` when using these metrics.

## ObjectKeypointSimilarity

For more information, see [object-keypoint-similarity](https://learnopencv.com/object-keypoint-similarity/).

**Params**

| Key                | Type                  | Default value | Description                                                                                                                                                         |
| ------------------ | --------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sigmas`           | `list[float] \| None` | `None`        | List of sigma values for each keypoint. If `None`, the COCO sigmas are used when the COCO dataset is provided. Otherwise, a default sigma value of 0.04 is applied. |
| `area_factor`      | `float`               | `0.53`        | Factor by which to multiply the bounding box area                                                                                                                   |
| `use_cocoeval_oks` | `bool`                | `True`        | Whether to use the same OKS formula as in COCO evaluation                                                                                                           |

## MeanAveragePrecision

Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` for object detection or instance segmentation predictions. This metric is built upon [pycocotools](https://github.com/cocodataset/cocoapi), which provides robust tools for evaluating these tasks.

```math
\text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i
```

where $AP_i$ is the average precision for class $i$ and $n$ is the number of classes. Average precision is the area under the precision-recall curve. For both detection and segmentation tasks, precision and recall are computed based on the Intersection over Union (IoU) between predicted and ground truth regions (bounding boxes or masks). A prediction is considered a true positive if its IoU exceeds a threshold $t$. Precision is defined as the ratio of true positives to all detections, and recall as the ratio of true positives to all ground truth instances.

## MeanAveragePrecisionKeypoints

Similar to [MeanAveragePrecision](#meanaverageprecision), but uses [OKS](#objectkeypointsimilarity) as `IoU` measure.
For a deeper understanding of how OKS works, please refer to the detailed explanation provided [here](https://learnopencv.com/object-keypoint-similarity/).
Evaluation leverages the [pycocotools](https://github.com/cocodataset/cocoapi) framework to assess mAP performance.

**Params**

| Key           | Type                                | Default value | Description                                                           |
| ------------- | ----------------------------------- | ------------- | --------------------------------------------------------------------- |
| `sigmas`      | `list[float] \| None`               | `None`        | List of sigmas for each keypoint. If `None`, the COCO sigmas are used |
| `area_factor` | `float`                             | `0.53`        | Factor by which to multiply the bounding box area                     |
| `max_dets`    | `int`                               | `20`          | Maximum number of detections per image                                |
| `box_fotmat`  | `Literal["xyxy", "xywh", "cxcywh"]` | `"xyxy"`      | Format of the bounding boxes                                          |

## ClosestIsPositiveAccuracy

Compute the accuracy of the closest positive sample to the query sample.
Needs to be connected to the `GhostFaceNetHead` node.

## MedianDistances

Compute the median distance between the query and the positive samples.
Needs to be connected to the `GhostFaceNetHead` node.

## OCRAccuracy

Accuracy metric for OCR tasks.

**Params**

| Key         | Type  | Default value | Description                                |
| ----------- | ----- | ------------- | ------------------------------------------ |
| `blank_cls` | `int` | `0`           | Index of the blank class. Defaults to `0`. |

## Confusion Matrix

Compute the confusion matrix for various tasks including classification, segmentation, and object detection.

> \[!NOTE\]
> **Important:** Both the Confusion Matrix (CM) and Mean Average Precision (mAP) metrics are sensitive to NMS parameters, such as confidence (`conf`) and IoU (`iout`) thresholds. Make sure to adjust these settings appropriately for your specific use case.
