# Metrics

List of all the available metrics.

## Table Of Contents

- [Torchmetrics](#torchmetrics)
- [ObjectKeypointSimilarity](#objectkeypointsimilarity)
- [MeanAveragePrecision](#meanaverageprecision)
- [MeanAveragePrecisionKeypoints](#meanaverageprecisionkeypoints)

## Torchmetrics

Metrics from the [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/) module.

- [Accuracy](https://lightning.ai/docs/torchmetrics/stable/classification/accuracy.html)
- [JaccardIndex](https://lightning.ai/docs/torchmetrics/stable/classification/jaccard_index.html) -- Intersection over Union.
- [F1Score](https://lightning.ai/docs/torchmetrics/stable/classification/f1_score.html)
- [Precision](https://lightning.ai/docs/torchmetrics/stable/classification/precision.html)
- [Recall](https://lightning.ai/docs/torchmetrics/stable/classification/recall.html)

## ObjectKeypointSimilarity

For more information, see [object-keypoint-similarity](https://learnopencv.com/object-keypoint-similarity/).

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
