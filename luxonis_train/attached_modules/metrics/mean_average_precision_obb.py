import numpy as np
import torch
from torch import Tensor

from luxonis_train.utils.boxutils import batch_probiou, xyxyxyxy2xywhr
from luxonis_train.utils.types import Labels, LabelType, Packet

from .base_metric import BaseMetric


class MeanAveragePrecisionOBB(BaseMetric):
    """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) for object
    detection predictions using oriented bounding boxes.

    Partially adapted from U{YOLOv8 OBBMetrics
    <https://github.com/ultralytics/ultralytics/blob/ba438aea5ae4d0e7c28d59ed8408955d16ca71ec/ultralytics/utils/metrics.py#L1223>}.
    """

    supported_labels = [LabelType.OBOUNDINGBOX]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.p = []  # precision for each class. Shape: (nc,)
        self.r = []  # recall for each class. Shape: (nc,)
        self.f1 = []  # F1 score for each class. Shape: (nc,)
        self.all_ap = []  # AP scores for all classes and all IoU thresholds. Shape: (nc, 10)
        self.ap_class_index = []  # index of class for each AP score. Shape: (nc,)
        self.nc = 0  # number of classes

        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])

        self.iouv = torch.linspace(
            0.5, 0.95, 10
        )  # IoU thresholds from 0.50 to 0.95 in spaces of 0.05 mAP@0.5:0.95

    def update(
        self,
        outputs: list[Tensor],  # preds
        labels: list[Tensor],  # batch
    ):
        """Update metrics without erasing stats from the previous batch, i.e. the
        metrics are calculated cumulatively.

        @type outputs: list[Tensor]
        @param outputs: Network predictions [x1, y1, x2, y2, conf, cls_idx, r]
            unnormalized (not in [0, 1] range) [Tensor(n_bboxes, 7)]
        @type labels: list[Tensor]
        @param labels: [cls_idx, x1, y1, x2, y2, r] unnormalized (not in [0, 1] range)
            [Tensor(n_bboxes, 6)]
        """
        for si, output in enumerate(outputs):
            self.stats["conf"].append(output[:, 4])
            self.stats["pred_cls"].append(output[:, 5])
            self.stats["target_cls"].append(labels[si][:, 0])
            gt_cls = labels[si][:, :1]  # cls_idx
            gt_bboxes = labels[si][:, 1:]  # [x1, y1, x2, y2, r]
            self.stats["tp"].append(
                self._process_batch(
                    detections=output, gt_bboxes=gt_bboxes, gt_cls=gt_cls
                )
            )

        results = self._process(
            torch.cat(self.stats["tp"]).cpu().numpy(),
            torch.cat(self.stats["conf"]).cpu().numpy(),
            torch.cat(self.stats["pred_cls"]).cpu().numpy(),
            torch.cat(self.stats["target_cls"]).cpu().numpy(),
        )

        self._update_metrics(results)

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[Tensor], list[Tensor]]:
        # outputs_nms: [x, y, w, h, r, conf, cls_idx] unnormalized (not in [0, 1] range) [Tensor(n_bboxes, 7)]
        # obb_labels: [img_id, cls_idx, x1, y1, x2, y2, x3, y3, x4, y4] normalized (in [0, 1] range) [Tensor(n_bboxes, 10)]
        obb_labels = self.get_label(labels)[0]
        output_nms = self.get_input_tensors(outputs)
        pred_scores = self.get_input_tensors(outputs, "class_scores")[
            0
        ]  # needed for batch size

        batch_size = pred_scores.shape[0]
        img_size = self.node.original_in_shape[1:]

        output_labels = []
        for i in range(len(output_nms)):
            output_nms[i][..., [0, 1, 2, 3, 4, 5, 6]] = output_nms[i][
                ..., [0, 1, 2, 3, 5, 6, 4]
            ]  # move angle to the end

            curr_label = obb_labels[obb_labels[:, 0] == i]
            output_labels.append(
                self._preprocess_target(curr_label, batch_size, img_size)
            )

        return output_nms, output_labels

    def _preprocess_target(self, target: Tensor, batch_size: int, img_size) -> Tensor:
        """Preprocess target in shape [batch_size, N, 6] where N is maximum number of
        instances in one image."""
        cls_idx = target[:, 1].unsqueeze(-1)
        xyxyxyxy = target[:, 2:]
        xyxyxyxy[:, 0::2] *= img_size[1]  # scale x
        xyxyxyxy[:, 1::2] *= img_size[0]  # scale y
        xcycwhr = xyxyxyxy2xywhr(xyxyxyxy)
        if isinstance(xcycwhr, np.ndarray):
            xcycwhr = torch.tensor(xcycwhr)
        out_target = torch.cat([cls_idx, xcycwhr], dim=-1)
        return out_target

    def reset(self) -> None:
        self.p = []
        self.r = []
        self.f1 = []
        self.all_ap = []
        self.ap_class_index = []

    def compute(
        self,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Process predicted results for object detection and update metrics."""
        results = self._process(
            torch.cat(self.stats["tp"]).cpu().numpy(),
            torch.cat(self.stats["conf"]).cpu().numpy(),
            torch.cat(self.stats["pred_cls"]).cpu().numpy(),
            torch.cat(self.stats["target_cls"]).cpu().numpy(),
        )

        metrics = {
            "p": torch.tensor(np.mean(results[0])),
            "r": torch.tensor(np.mean(results[1])),
            "f1": torch.tensor(np.mean(results[2])),
            "all_ap": torch.tensor(np.mean(results[3])),
            "ap_class_index": torch.tensor(np.mean(results[4])),
        }

        map = torch.tensor(MeanAveragePrecisionOBB.map(results[5]))  # all_ap

        return map, metrics

    def _process_batch(
        self, detections: Tensor, gt_bboxes: Tensor, gt_cls: Tensor
    ) -> Tensor:
        """Perform computation of the correct prediction matrix for a batch of # "fp":
        torch.from_numpy(results[1]), detections and ground truth bounding boxes.

        @type detections: Tensor
        @param detections: A tensor of shape (N, 7) representing the detected bounding boxes and associated
            data. Each detection is represented as (x1, y1, x2, y2, conf, class, angle).
        @type gt_bboxes: Tensor
        @param gt_bboxes: A tensor of shape (M, 5) representing the ground truth bounding boxes. Each box is
            represented as (x1, y1, x2, y2, angle).
        @type gt_cls: Tensor
        @param gt_cls: A tensor of shape (M,) representing class labels for the ground truth bounding boxes.
        @rtype: Tensor
        @return: The correct prediction matrix with shape (N, 10), which includes 10 IoU (Intersection over
            Union) levels for each detection, indicating the accuracy of predictions compared to the ground truth.

        Example:
            ```python
            detections = torch.rand(100, 7)  # 100 sample detections
            gt_bboxes = torch.rand(50, 5)  # 50 sample ground truth boxes
            gt_cls = torch.randint(0, 5, (50,))  # 50 ground truth class labels
            correct_matrix = OBBValidator._process_batch(detections, gt_bboxes, gt_cls)
            ```

        Note:
            This method relies on `batch_probiou` to calculate IoU between detections and ground truth bounding boxes.
        """
        iou = batch_probiou(
            gt_bboxes,
            torch.cat([detections[:, :4], detections[:, -1:]], dim=-1),
        )
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def match_predictions(
        self,
        pred_classes: Tensor,
        true_classes: Tensor,
        iou: Tensor,
        use_scipy: bool = False,
    ) -> Tensor:
        """Matches predictions to ground truth objects (pred_classes, true_classes)
        using IoU.

        @type pred_classes: Tensor
        @param pred_classes: Predicted class indices of shape(N,).
        @type true_classes: Tensor
        @param true_classes: Target class indices of shape(M,).
        @type iou: Tensor
        @param iou: An NxM tensor containing the pairwise IoU values for predictions and
            ground of truth
        @type use_scipy: bool
        @param use_scipy: Whether to use scipy for matching (more precise).
        @rtype: Tensor
        @return: Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(
                        cost_matrix, maximize=True
                    )
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(
                    iou >= threshold
                )  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[
                            iou[matches[:, 0], matches[:, 1]].argsort()[::-1]
                        ]
                        matches = matches[
                            np.unique(matches[:, 1], return_index=True)[1]
                        ]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[
                            np.unique(matches[:, 0], return_index=True)[1]
                        ]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def _update_metrics(self, results: tuple[np.ndarray, ...]):
        """Updates the evaluation metrics of the model with a new set of results.

        @type results: tuple[np.ndarray, ...]
        @param results: A tuple containing the following evaluation metrics:
            - p (list): Precision for each class. Shape: (nc,).
            - r (list): Recall for each class. Shape: (nc,).
            - f1 (list): F1 score for each class. Shape: (nc,).
            - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
            - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        # The following logic impies averaging AP over all classes
        self.p = torch.tensor(np.mean(results[0]))
        self.r = torch.tensor(np.mean(results[1]))
        self.f1 = torch.tensor(np.mean(results[2]))
        self.all_ap = torch.tensor(np.mean(results[3]))
        self.ap_class_index = torch.tensor(np.mean(results[4]))
        # (
        #     self.p,
        #     self.r,
        #     self.f1,
        #     self.all_ap,
        #     self.ap_class_index,
        #     _,  # self.p_curve,
        #     _,  # self.r_curve,
        #     _,  # self.f1_curve,
        #     _,  # self.px,
        #     _,  # self.prec_values,
        # ) = results

    def _process(
        self,
        tp: np.ndarray,
        conf: np.ndarray,
        pred_cls: np.ndarray,
        target_cls: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        """Process predicted results for object detection and update metrics."""
        results = MeanAveragePrecisionOBB.ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            # plot=self.plot,
            # save_dir=self.save_dir,
            # names=self.names,
            # on_plot=self.on_plot,
        )[2:]
        return results

    @staticmethod
    def ap_per_class(
        tp: np.ndarray,
        conf: np.ndarray,
        pred_cls: np.ndarray,
        target_cls: np.ndarray,
        # plot=False,
        # on_plot=None,
        # save_dir=Path(),
        # names={},
        eps: float = 1e-16,
        # prefix="",
    ) -> tuple[np.ndarray, ...]:
        """Computes the average precision per class for object detection evaluation.

        Args:
            tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
            conf (np.ndarray): Array of confidence scores of the detections.
            pred_cls (np.ndarray): Array of predicted classes of the detections.
            target_cls (np.ndarray): Array of true classes of the detections.
            plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
            on_plot (func, optional): A callback to pass plots path and data when they are rendered. Defaults to None.
            save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
            names (dict, optional): Dict of class names to plot PR curves. Defaults to an empty tuple.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
            prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

        Returns:
            (tuple): A tuple of six arrays and one array of unique classes, where:
                tp (np.ndarray): True positive counts at threshold given by max F1 metric for each class.Shape: (nc,).
                fp (np.ndarray): False positive counts at threshold given by max F1 metric for each class. Shape: (nc,).
                p (np.ndarray): Precision values at threshold given by max F1 metric for each class. Shape: (nc,).
                r (np.ndarray): Recall values at threshold given by max F1 metric for each class. Shape: (nc,).
                f1 (np.ndarray): F1-score values at threshold given by max F1 metric for each class. Shape: (nc,).
                ap (np.ndarray): Average precision for each class at different IoU thresholds. Shape: (nc, 10).
                unique_classes (np.ndarray): An array of unique classes that have data. Shape: (nc,).
                p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
                r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
                f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
                x (np.ndarray): X-axis values for the curves. Shape: (1000,).
                prec_values: Precision values at mAP@0.5 for each class. Shape: (nc, 1000).
        """
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique classes
        unique_classes, nt = np.unique(target_cls, return_counts=True)
        nc = unique_classes.shape[0]  # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        x, prec_values = np.linspace(0, 1, 1000), []

        # Average precision, precision and recall curves
        ap, p_curve, r_curve = (
            np.zeros((nc, tp.shape[1])),
            np.zeros((nc, 1000)),
            np.zeros((nc, 1000)),
        )
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = nt[ci]  # number of labels
            n_p = i.sum()  # number of predictions
            if n_p == 0 or n_l == 0:
                continue

            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + eps)  # recall curve
            r_curve[ci] = np.interp(
                -x, -conf[i], recall[:, 0], left=0
            )  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p_curve[ci] = np.interp(
                -x, -conf[i], precision[:, 0], left=1
            )  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = MeanAveragePrecisionOBB.compute_ap(
                    recall[:, j], precision[:, j]
                )
                # if plot and j == 0:
                #     prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)

        # Compute F1 (harmonic mean of precision and recall)
        f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)

        i = MeanAveragePrecisionOBB.smooth(
            f1_curve.mean(0), 0.1
        ).argmax()  # max F1 index
        p, r, f1 = (
            p_curve[:, i],
            r_curve[:, i],
            f1_curve[:, i],
        )  # max-F1 precision, recall, F1 values
        tp = (r * nt).round()  # true positives
        fp = (tp / (p + eps) - tp).round()  # false positives
        return (
            tp,
            fp,
            p,
            r,
            f1,
            ap,
            unique_classes.astype(int),
            p_curve,
            r_curve,
            f1_curve,
            x,
            prec_values,
        )

    @staticmethod
    def compute_ap(
        recall: list[float], precision: list[float]
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute the average precision (AP) given the recall and precision curves.

        Args:
            recall (list): The recall curve.
            precision (list): The precision curve.

        Returns:
            (float): Average precision.
            (np.ndarray): Precision envelope curve.
            (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
        """
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))

        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = "interp"  # methods: 'continuous', 'interp'
        if method == "interp":
            x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
        else:  # 'continuous'
            i = np.where(mrec[1:] != mrec[:-1])[
                0
            ]  # points where x-axis (recall) changes
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

        return ap, mpre, mrec

    @staticmethod
    def smooth(y: np.ndarray, f: float = 0.05) -> np.ndarray:
        """Box filter of fraction f."""
        nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
        p = np.ones(nf // 2)  # ones padding
        yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
        return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed

    @staticmethod
    def map(all_ap: np.ndarray) -> float:
        """
        Returns the mean Average Precision (mAP) over IoU thresholds of 0.5 - 0.95 in steps of 0.05.

        Returns:
            (float): The mAP over IoU thresholds of 0.5 - 0.95 in steps of 0.05.
        """
        return all_ap.mean() if len(all_ap) else 0.0
