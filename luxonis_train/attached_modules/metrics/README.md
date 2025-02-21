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

Compute the `Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)` for object detection predictions.

```math
\text{mAP} = \frac{1}{n} \sum_{i=1}^{n} AP_i
```

where $AP_i$ is the average precision for class $i$ and $n$ is the number of classes. The average
precision is defined as the area under the precision-recall curve. For object detection the recall and precision are
defined based on the intersection of union (IoU) between the predicted bounding boxes and the ground truth bounding
boxes e.g. if two boxes have an IoU > t (with t being some threshold) they are considered a match and therefore
considered a true positive. The precision is then defined as the number of true positives divided by the number of
all detected boxes and the recall is defined as the number of true positives divided by the number of all ground
boxes.

## MeanAveragePrecisionKeypoints

Similar to [MeanAveragePrecision](#meanaverageprecision), but uses [OKS](#objectkeypointsimilarity) as `IoU` measure.
For a deeper understanding of how OKS works, please refer to the detailed explanation provided [here](https://learnopencv.com/object-keypoint-similarity/).
Evaluation leverages COCO evaluation framework (COCOeval) to assess mAP performance.

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
