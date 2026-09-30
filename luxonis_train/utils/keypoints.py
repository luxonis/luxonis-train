"""Math helpers for keypoints.

The module holds the keypoint sigmas, the object keypoint similarity,
and the conversions between keypoints and bounding boxes.

"""

import torch
from loguru import logger
from torch import Tensor


def get_sigmas(
    sigmas: list[float] | None,
    n_keypoints: int,
    caller_name: str | None = None,
) -> Tensor:
    """Validate the given keypoint sigmas or create the default ones.

    The sigmas are the per-keypoint scales of the object keypoint
    similarity in `compute_pose_oks`. When ``sigmas`` is ``None``, the
    function selects the defaults:

    - For ``17`` keypoints, it returns the COCO person sigmas and logs a
      warning.
    - For any other count, it returns ``0.04`` for each keypoint and
      logs an info message.

    Args:
        sigmas (list[float] | None): One sigma per keypoint. ``None``
            selects the defaults.
        n_keypoints (int): The number of keypoints.
        caller_name (str | None): The name of the caller, used as a
            prefix of the log and error messages. ``None`` adds no
            prefix.

    Returns:
        ``Tensor``: The sigmas as a ``float32`` tensor of shape
        ``[n_keypoints]``.

    Raises:
        ValueError: When ``sigmas`` is given and its length differs from
            ``n_keypoints``.

    Examples:
        >>> get_sigmas([0.1, 0.2], 2).shape
        torch.Size([2])

        >>> get_sigmas([0.1], 2)
        Traceback (most recent call last):
        ValueError: The length of the sigmas list must be the same ...

    """
    if sigmas is not None:
        if len(sigmas) == n_keypoints:
            return torch.tensor(sigmas, dtype=torch.float32)
        error_msg = "The length of the sigmas list must be the same as the number of keypoints."
        if caller_name:
            error_msg = f"[{caller_name}] {error_msg}"
        raise ValueError(error_msg)
    if n_keypoints == 17:
        msg = "Default COCO sigmas are being used."
        if caller_name:
            msg = f"[{caller_name}] {msg}"
        logger.warning(msg)
        return torch.tensor(
            [
                0.026,
                0.025,
                0.025,
                0.035,
                0.035,
                0.079,
                0.079,
                0.072,
                0.072,
                0.062,
                0.062,
                0.107,
                0.107,
                0.087,
                0.087,
                0.089,
                0.089,
            ],
            dtype=torch.float32,
        )
    msg = "Default sigma of 0.04 is being used for each keypoint."
    if caller_name:
        msg = f"[{caller_name}] {msg}"
    logger.info(msg)
    return torch.tensor([0.04] * n_keypoints, dtype=torch.float32)


def get_center_keypoints(
    bboxes: Tensor, *, height: int = 1, width: int = 1
) -> Tensor:
    """Make one center keypoint per bounding box.

    The FOMO loss and the object keypoint similarity metric use the box
    centers as the keypoint targets of the FOMO task. That task has no
    annotated keypoints.

    Args:
        bboxes (``Tensor``): The bounding boxes of shape ``[N, 6]``, with
            the columns ``(batch_index, class, x, y, w, h)``. ``x`` and
            ``y`` are the normalized top-left corner, and ``w`` and
            ``h`` are the normalized size.
        height (int): The height that scales ``y``, for example the
            height of an image or of a heatmap. ``1`` keeps the
            coordinates normalized.
        width (int): The width that scales ``x``, for example the width
            of an image or of a heatmap. ``1`` keeps the coordinates
            normalized.

    Returns:
        ``Tensor``: The keypoints of shape ``[N, 4]``, with the columns
        ``(batch_index, x, y, visibility)``, on the device and of the
        dtype of ``bboxes``. ``x`` and ``y`` are the box center scaled
        by ``width`` and ``height``. The visibility is always ``2``.

    Example:
        >>> import torch
        >>> bboxes = torch.tensor([[0.0, 1.0, 0.25, 0.5, 0.5, 0.25]])
        >>> get_center_keypoints(bboxes).tolist()
        [[0.0, 0.5, 0.625, 2.0]]
        >>> get_center_keypoints(bboxes, height=200, width=100).tolist()
        [[0.0, 50.0, 125.0, 2.0]]

    """
    keypoints = torch.full(
        (bboxes.shape[0], 4), 2, device=bboxes.device, dtype=bboxes.dtype
    )
    keypoints[:, 0] = bboxes[:, 0]
    keypoints[:, 1] = (bboxes[:, 2] + bboxes[:, 4] / 2) * width
    keypoints[:, 2] = (bboxes[:, 3] + bboxes[:, 5] / 2) * height
    return keypoints


def insert_class(keypoints: Tensor, bboxes: Tensor) -> Tensor:
    """Insert the class index of each bounding box into its keypoints.

    Args:
        keypoints (``Tensor``): The keypoints of shape ``[N, 1 + 3K]``,
            where ``K`` is the number of keypoints. The batch index is
            in the first column, followed by ``(x, y, visibility)``
            triples.
        bboxes (``Tensor``): The bounding boxes of shape ``[N, 6]``, in
            the same instance order, with the class index in the
            second column.

    Returns:
        ``Tensor``: The keypoints of shape ``[N, 2 + 3K]``, with the
        class index inserted as the second column.

    Example:
        >>> import torch
        >>> keypoints = torch.tensor([[0.0, 0.5, 0.5, 2.0]])
        >>> bboxes = torch.tensor([[0.0, 3.0, 0.1, 0.1, 0.2, 0.2]])
        >>> insert_class(keypoints, bboxes).tolist()
        [[0.0, 3.0, 0.5, 0.5, 2.0]]

    """
    classes = bboxes[:, 1]
    return torch.cat(
        (
            keypoints[:, :1],
            classes.unsqueeze(-1),
            keypoints[:, 1:],
        ),
        dim=-1,
    )


def compute_pose_oks(
    predictions: Tensor,
    targets: Tensor,
    sigmas: Tensor,
    gt_bboxes: Tensor | None = None,
    pose_area: Tensor | None = None,
    eps: float = 1e-9,
    area_factor: float = 0.53,
    use_cocoeval_oks: bool = True,
) -> Tensor:
    r"""Compute the object keypoint similarity of each target-prediction pair.

    For one image, the similarity of target :math:`t` and prediction
    :math:`p` is the mean of this term over the visible keypoints of
    :math:`t`:

    .. math::

        \exp\left(-\frac{d_i^2}{2 \, (2 \sigma_i)^2 \, A}\right)

    In the term, :math:`d_i` is the distance between the two keypoints
    :math:`i`, and :math:`\sigma_i` is the sigma of keypoint :math:`i`.
    :math:`A` is the pose area of :math:`t`. A target keypoint is
    visible when its third value is greater than ``0``. A target
    without visible keypoints gets a similarity of ``0``. With
    ``use_cocoeval_oks`` set to ``False``, the exponent is
    :math:`-d_i^2 / (2 (A \sigma_i)^2)` instead.

    Args:
        predictions (``Tensor``): The predicted keypoints of shape
            ``[N, M2, n_keypoints, 3]``. The function reads only ``x``
            and ``y``, the first two values of each keypoint.
        targets (``Tensor``): The target keypoints of shape
            ``[N, M1, n_keypoints, 3]``, as ``(x, y, visibility)``.
        sigmas (``Tensor``): One sigma per keypoint, of shape
            ``[n_keypoints]``.
        gt_bboxes (``Tensor | None``): The target boxes of shape
            ``[N, M1, 4]`` in ``xyxy`` format. Their area times
            ``area_factor`` is the pose area. The function reads them
            only when ``pose_area`` is ``None``.
        pose_area (``Tensor | None``): The pose area of each target, of
            shape ``[N, M1, 1, 1]``. ``None`` computes it from
            ``gt_bboxes``.
        eps (float): A small constant that the function adds to the
            area and to the visible count. It prevents a division by
            zero.
        area_factor (float): The factor that scales the box area to the
            pose area.
        use_cocoeval_oks (bool): When ``True``, use the formula of the
            COCO evaluation code. When ``False``, use the other formula
            above.

    Returns:
        ``Tensor``: The similarities of shape ``[N, M1, M2]``, in
        ``[0, 1]``.

    Raises:
        ValueError: When both ``pose_area`` and ``gt_bboxes`` are
            ``None``.

    References:
        - COCO keypoint evaluation: https://cocodataset.org/#keypoints-eval
        - ``computeOks`` in ``pycocotools/cocoeval.py``:
          https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229

    Examples:
        >>> import torch
        >>> targets = torch.tensor([[[[0.5, 0.5, 2.0]]]])
        >>> sigmas = torch.tensor([0.05])
        >>> gt_bboxes = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
        >>> exact = torch.tensor([[[[0.5, 0.5, 1.0]]]])
        >>> oks = compute_pose_oks(exact, targets, sigmas, gt_bboxes=gt_bboxes)
        >>> oks.round(decimals=3).tolist()
        [[[1.0]]]

        >>> far = torch.tensor([[[[5.0, 5.0, 1.0]]]])
        >>> oks = compute_pose_oks(far, targets, sigmas, gt_bboxes=gt_bboxes)
        >>> oks.round(decimals=3).tolist()
        [[[0.0]]]

    """
    if pose_area is None:
        if gt_bboxes is None:
            raise ValueError(
                "Either 'pose_area' or 'gt_bboxes' must be provided."
            )
        width = gt_bboxes[:, :, 2] - gt_bboxes[:, :, 0]
        height = gt_bboxes[:, :, 3] - gt_bboxes[:, :, 1]
        pose_area = (
            (width * height * area_factor).unsqueeze(-1).unsqueeze(-1)
        )  # shape: [N, M1, 1, 1]

    gt_xy = targets[:, :, :, :2].unsqueeze(
        2
    )  # shape: [N, M1, 1, n_keypoints, 2]
    pred_xy = predictions[:, :, :, :2].unsqueeze(
        1
    )  # shape: [N, 1, M2, n_keypoints, 2]

    sq_diff = ((gt_xy - pred_xy) ** 2).sum(
        dim=-1
    )  # shape: [N, M1, M2, n_keypoints]

    sigmas = sigmas.view(1, 1, 1, -1)  # shape: [1, 1, 1, n_keypoints]

    if use_cocoeval_oks:
        # use same formula as in COCOEval script here:
        # https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py#L229
        exp_term = sq_diff / ((2 * sigmas) ** 2) / (pose_area + eps) / 2
    else:
        # use same formula as defined here: https://cocodataset.org/#keypoints-eval
        exp_term = sq_diff / ((pose_area + eps) * sigmas) ** 2 / 2

    oks_vals = torch.exp(-exp_term)  # shape: [N, M1, M2, n_keypoints]

    vis_mask = (
        targets[:, :, :, 2].gt(0).float().unsqueeze(2)
    )  # shape: [N, M1, 1, n_keypoints]
    vis_count = vis_mask.sum(dim=-1)  # shape: [N, M1, 1]

    return (oks_vals * vis_mask).sum(dim=-1) / (
        vis_count + eps
    )  # shape: [N, M1, M2]
