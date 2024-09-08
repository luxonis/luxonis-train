import numpy as np
import torch
from torch import Tensor
from torchvision.ops import box_convert

from luxonis_train.utils.boxutils import batch_probiou
from luxonis_train.utils.types import Labels, LabelType, Packet

from .base_metric import BaseMetric


class MeanAveragePrecisionOBB(BaseMetric):
    """Compute the Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR) for
    oriented object detection predictions.

    Adapted from U{Mean-Average-Precision (mAP) and Mean-Average-Recall (mAR)
    <https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html>}.
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

        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def update(
        self,
        preds: list[dict[str, Tensor]],  # outputs
        batch: list[dict[str, Tensor]],  # labels
    ):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(
                            detections=None, gt_bboxes=bbox, gt_cls=cls
                        )
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # # Save
            # if self.args.save_json:
            #     self.pred_to_json(predn, batch["im_file"][si])
            # if self.args.save_txt:
            #     self.save_one_txt(
            #         predn,
            #         self.args.save_conf,
            #         pbatch["ori_shape"],
            #         self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
            #     )

    def prepare(
        self, outputs: Packet[Tensor], labels: Labels
    ) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        box_label = self.get_label(labels)[0]
        output_nms = self.get_input_tensors(outputs)

        image_size = self.node.original_in_shape[1:]

        output_list: list[dict[str, Tensor]] = []
        label_list: list[dict[str, Tensor]] = []
        for i in range(len(output_nms)):
            output_list.append(
                {
                    "boxes": output_nms[i][:, :4],
                    "scores": output_nms[i][:, 4],
                    "labels": output_nms[i][:, 5].int(),
                }
            )

            curr_label = box_label[box_label[:, 0] == i]
            curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
            curr_bboxs[:, 0::2] *= image_size[1]
            curr_bboxs[:, 1::2] *= image_size[0]
            label_list.append({"boxes": curr_bboxs, "labels": curr_label[:, 1].int()})

        return output_list, label_list

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(
                torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            )  # target boxes
            # ops.scale_boxes(
            #     imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True
            # )  # native-space labels
        return {
            "cls": cls,
            "bbox": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
        }

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for OBB validation with scaled and padded
        bounding boxes."""
        predn = pred.clone()
        # ops.scale_boxes(
        #     pbatch["imgsz"],
        #     predn[:, :4],
        #     pbatch["ori_shape"],
        #     ratio_pad=pbatch["ratio_pad"],
        #     xywh=True,
        # )  # native-space pred
        return predn

    def reset(self) -> None:
        self.metric.reset()

    def compute(self) -> tuple[Tensor, dict[str, Tensor]]:
        pass
        # metric_dict = self.metric.compute()

        # del metric_dict["classes"]
        # del metric_dict["map_per_class"]
        # del metric_dict["mar_100_per_class"]
        # map = metric_dict.pop("map")

        # mat = self._process_batch()

        # return map, metric_dict

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """Perform computation of the correct prediction matrix for a batch of
        detections and ground truth bounding boxes.

        Args:
            detections (torch.Tensor): A tensor of shape (N, 7) representing the detected bounding boxes and associated
                data. Each detection is represented as (x1, y1, x2, y2, conf, class, angle).
            gt_bboxes (torch.Tensor): A tensor of shape (M, 5) representing the ground truth bounding boxes. Each box is
                represented as (x1, y1, x2, y2, angle).
            gt_cls (torch.Tensor): A tensor of shape (M,) representing class labels for the ground truth bounding boxes.

        Returns:
            (torch.Tensor): The correct prediction matrix with shape (N, 10), which includes 10 IoU (Intersection over
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
            gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1)
        )
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def match_predictions(self, pred_classes, true_classes, iou, use_scipy=False):
        """Matches predictions to ground truth objects (pred_classes, true_classes)
        using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
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

    def update_metric(self, results):
        """Updates the evaluation metrics of the model with a new set of results.

        Args:
            results (tuple): A tuple containing the following evaluation metrics:
                - p (list): Precision for each class. Shape: (nc,).
                - r (list): Recall for each class. Shape: (nc,).
                - f1 (list): F1 score for each class. Shape: (nc,).
                - all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
                - ap_class_index (list): Index of class for each AP score. Shape: (nc,).

        Side Effects:
            Updates the class attributes `self.p`, `self.r`, `self.f1`, `self.all_ap`, and `self.ap_class_index` based
            on the values provided in the `results` tuple.
        """
        (
            self.p,
            self.r,
            self.f1,
            self.all_ap,
            self.ap_class_index,
            self.p_curve,
            self.r_curve,
            self.f1_curve,
            self.px,
            self.prec_values,
        ) = results

    def process(self, tp, conf, pred_cls, target_cls):
        """Process predicted results for object detection and update metrics."""
        results = MeanAveragePrecisionOBB.ap_per_class(
            tp,
            conf,
            pred_cls,
            target_cls,
            # plot=self.plot,
            # save_dir=self.save_dir,
            # names=self.names,
            on_plot=self.on_plot,
        )[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @staticmethod
    def ap_per_class(
        tp,
        conf,
        pred_cls,
        target_cls,
        plot=False,
        on_plot=None,
        # save_dir=Path(),
        # names={},
        eps=1e-16,
        prefix="",
    ):
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
            # for j in range(tp.shape[1]):
            #     ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            #     if plot and j == 0:
            #         prec_values.append(np.interp(x, mrec, mpre))  # precision at mAP@0.5

        prec_values = np.array(prec_values)  # (nc, 1000)

        # Compute F1 (harmonic mean of precision and recall)
        # f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
        # names = [
        #     v for k, v in names.items() if k in unique_classes
        # ]  # list: only classes that have data
        # names = dict(enumerate(names))  # to dict
        # if plot:
        #     plot_pr_curve(
        #         x,
        #         prec_values,
        #         ap,
        #         save_dir / f"{prefix}PR_curve.png",
        #         names,
        #         on_plot=on_plot,
        #     )
        #     plot_mc_curve(
        #         x,
        #         f1_curve,
        #         save_dir / f"{prefix}F1_curve.png",
        #         names,
        #         ylabel="F1",
        #         on_plot=on_plot,
        #     )
        #     plot_mc_curve(
        #         x,
        #         p_curve,
        #         save_dir / f"{prefix}P_curve.png",
        #         names,
        #         ylabel="Precision",
        #         on_plot=on_plot,
        #     )
        #     plot_mc_curve(
        #         x,
        #         r_curve,
        #         save_dir / f"{prefix}R_curve.png",
        #         names,
        #         ylabel="Recall",
        #         on_plot=on_plot,
        #     )

        # i = smooth(f1_curve.mean(0), 0.1).argmax()  # max F1 index
        # p, r, f1 = (
        #     p_curve[:, i],
        #     r_curve[:, i],
        #     f1_curve[:, i],
        # )  # max-F1 precision, recall, F1 values
        # tp = (r * nt).round()  # true positives
        # fp = (tp / (p + eps) - tp).round()  # false positives
        return (
            # tp,
            # fp,
            # p,
            # r,
            # f1,
            ap,
            unique_classes.astype(int),
            p_curve,
            r_curve,
            # f1_curve,
            x,
            prec_values,
        )
