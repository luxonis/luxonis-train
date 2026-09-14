"""Helpers for the visualizers.

The helpers convert between tensors and images, denormalize the input
images, select colors and font sizes, draw labels, and combine the label
image and the prediction image into one image. ``Color`` is the type of
a color: a color name or a hex string such as ``"#FF0000"``, or an RGB
tuple.

"""

import colorsys
import io
from collections.abc import Mapping
from typing import Literal, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor
from torchvision.ops import box_convert
from torchvision.utils import (
    draw_bounding_boxes,
    draw_keypoints,
    draw_segmentation_masks,
)

from luxonis_train.config import Config

Color = str | tuple[int, int, int]


def get_prediction_labels(
    prediction: Tensor,
    label_dict: Mapping[int, str] | None,
    draw_labels: bool,
    draw_scores: bool,
) -> list[str] | None:
    """Build the text label of each predicted box.

    A label holds the class name when ``draw_labels`` is set, and the
    confidence with two decimals when ``draw_scores`` is set. A space
    separates the two parts.

    Args:
        prediction (``Tensor``): Boxes of shape ``[M, 6]``, with rows
            ``[x1, y1, x2, y2, conf, class]``.
        label_dict (``Mapping[int, str] | None``): Class names by class
            index. A class without a name, or every class when ``None``,
            gets its index as the name.
        draw_labels (bool): Whether the labels hold the class names.
        draw_scores (bool): Whether the labels hold the confidences.

    Returns:
        list[str] | None: One label for each box, or ``None`` when both
        ``draw_labels`` and ``draw_scores`` are ``False``.

    Example:
        >>> import torch
        >>> prediction = torch.tensor(
        ...     [
        ...         [0.0, 0.0, 4.0, 4.0, 0.75, 1.0],
        ...         [1.0, 1.0, 2.0, 2.0, 0.5, 7.0],
        ...     ]
        ... )
        >>> get_prediction_labels(prediction, {1: "cat"}, True, True)
        ['cat 0.75', '7 0.50']
        >>> get_prediction_labels(prediction, None, False, True)
        ['0.75', '0.50']

    """
    if not (draw_labels or draw_scores):
        return None

    prediction_classes = prediction[..., 5].int()
    prediction_scores = prediction[..., 4]

    labels: list[str] = []
    for score, class_id in zip(
        prediction_scores, prediction_classes, strict=True
    ):
        parts: list[str] = []
        if draw_labels:
            if label_dict is not None:
                parts.append(label_dict.get(int(class_id), str(int(class_id))))
            else:
                parts.append(str(int(class_id)))
        if draw_scores:
            parts.append(f"{float(score):.2f}")
        labels.append(" ".join(parts))

    return labels


def figure_to_torch(fig: Figure, width: int, height: int) -> Tensor:
    """Render a matplotlib figure to an RGB image tensor and close it.

    The function saves the figure as a PNG image with a tight bounding
    box and no padding, and resizes the image to ``width`` and
    ``height``. It then closes the figure with ``plt.close``.

    Args:
        fig (``Figure``): The matplotlib figure to render.
        width (int): The width of the image, in pixels.
        height (int): The height of the image, in pixels.

    Returns:
        ``Tensor``: A ``uint8`` image of shape ``[3, height, width]``.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> _ = ax.plot([0, 1], [0, 1])
        >>> figure_to_torch(fig, width=64, height=32).shape
        torch.Size([3, 32, 64])

    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = Image.open(buf).convert("RGB")
    img_arr = img_arr.resize((width, height))
    img_tensor = torch.tensor(np.array(img_arr)).permute(2, 0, 1)
    buf.close()
    plt.close(fig)
    return img_tensor


def torch_img_to_numpy(
    img: Tensor, reverse_colors: bool = False
) -> npt.NDArray[np.uint8]:
    """Convert a torch image to a ``uint8`` NumPy image.

    The function multiplies a floating point image by ``255`` and
    truncates it to integers. It clips all values to ``[0, 255]``.

    Args:
        img (``Tensor``): An image of shape ``[C, H, W]``. A floating
            point image has values in ``[0, 1]``.
        reverse_colors (bool): Whether to swap the first and the third
            channel, for example from RGB to BGR. The image must then
            have three channels.

    Returns:
        ``npt.NDArray[np.uint8]``: A new contiguous image of shape
        ``[H, W, C]``.

    Example:
        >>> import torch
        >>> img = torch.tensor([[[0.5]], [[1.0]], [[2.0]]])
        >>> torch_img_to_numpy(img).tolist()
        [[[127, 255, 255]]]
        >>> torch_img_to_numpy(img, reverse_colors=True).tolist()
        [[[255, 255, 127]]]

    """
    if img.is_floating_point():
        img = img.mul(255).int()
    img = torch.clamp(img, 0, 255)
    arr = img.detach().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    arr = np.ascontiguousarray(arr)
    if reverse_colors:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return cast(npt.NDArray[np.uint8], arr)


def numpy_to_torch_img(img: np.ndarray) -> Tensor:
    """Convert a NumPy image to a torch image.

    The result shares its memory with ``img`` and keeps its dtype.

    Args:
        img (``np.ndarray``): An image of shape ``[H, W, C]``.

    Returns:
        ``Tensor``: A view of shape ``[C, H, W]``.

    Example:
        >>> import numpy as np
        >>> numpy_to_torch_img(np.zeros((4, 6, 3), dtype=np.uint8)).shape
        torch.Size([3, 4, 6])

    """
    return torch.from_numpy(img).permute(2, 0, 1)


def preprocess_images(
    imgs: Tensor,
    mean: list[float] | float | None = None,
    std: list[float] | float | None = None,
) -> Tensor:
    """Convert a batch of model input images to ``uint8`` images.

    When ``mean`` or ``std`` is given, `denormalize` restores each image
    and scales it to ``[0, 255]``. Otherwise the function only casts the
    images to ``uint8``, so they must already have values in
    ``[0, 255]``.

    Args:
        imgs (``Tensor``): Images of shape ``[B, C, H, W]``.
        mean (list[float] | float | None): The mean of the normalization,
            one value for each channel or one value for all channels.
        std (list[float] | float | None): The standard deviation of the
            normalization, in the same form as ``mean``.

    Returns:
        ``Tensor``: A new ``uint8`` tensor of shape ``[B, C, H, W]``.

    Example:
        >>> import torch
        >>> imgs = torch.zeros(2, 3, 4, 4)
        >>> out = preprocess_images(imgs, mean=[0.5, 0.5, 0.5], std=0.5)
        >>> out.dtype, out[0, :, 0, 0].tolist()
        (torch.uint8, [127, 127, 127])

    """
    out_imgs = []
    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        if mean is not None or std is not None:
            curr_img = denormalize(curr_img, to_uint8=True, mean=mean, std=std)
        else:
            curr_img = curr_img.to(torch.uint8)

        out_imgs.append(curr_img)

    return torch.stack(out_imgs)


def draw_segmentation_targets(
    image: Tensor,
    target: Tensor,
    alpha: float = 0.4,
    colors: Color | list[Color] | None = None,
) -> Tensor:
    """Blend segmentation masks into an image.

    The function moves the image and the masks to the CPU and calls
    ``torchvision.utils.draw_segmentation_masks``. A pixel in more than
    one mask gets the color black before the blend.

    Args:
        image (``Tensor``): An RGB image of shape ``[3, H, W]``, of dtype
            ``uint8``, or floating point with values in ``[0, 1]``.
        target (``Tensor``): Masks of shape ``[N, H, W]``, or one mask of
            shape ``[H, W]``. Every non-zero value marks a pixel of the
            mask.
        alpha (float): The opacity of the masks, from ``0`` for
            transparent to ``1`` for opaque.
        colors (``Color | list[Color] | None``): One color for each
            mask, or one color for all masks. When ``None``,
            ``torchvision`` selects the colors.

    Returns:
        ``Tensor``: A new image on the CPU, of the same shape and dtype
        as ``image``. When ``target`` holds no masks, ``torchvision``
        issues a warning, and the function returns the image unchanged.

    Example:
        >>> import torch
        >>> image = torch.zeros(3, 1, 3, dtype=torch.uint8)
        >>> masks = torch.tensor([[[1, 1, 0]], [[0, 1, 1]]])
        >>> out = draw_segmentation_targets(
        ...     image, masks, alpha=1.0, colors=["red", "blue"]
        ... )
        >>> out[0].tolist(), out[2].tolist()
        ([[255, 0, 0]], [[0, 0, 255]])

    """
    masks = target.bool()
    masks = masks.cpu()
    image = image.cpu()
    return draw_segmentation_masks(image, masks, alpha=alpha, colors=colors)


def draw_bounding_box_labels(img: Tensor, label: Tensor, **kwargs) -> Tensor:
    """Draw normalized bounding boxes on an image.

    The function converts the boxes from normalized ``xywh``, where
    ``x`` and ``y`` are the top-left corner, to pixel ``xyxy`` with the
    image size. It then calls ``torchvision.utils.draw_bounding_boxes``.
    It does not change ``label``.

    Args:
        img (``Tensor``): A ``uint8`` image of shape ``[C, H, W]``.
        label (``Tensor``): Boxes of shape ``[N, 4]``, with rows
            ``[x, y, w, h]`` normalized to ``[0, 1]``.
        **kwargs (``Any``): Keyword arguments forwarded to
            ``draw_bounding_boxes``, such as ``labels``, ``colors``, and
            ``width``.

    Returns:
        ``Tensor``: A new image of the same shape as ``img``, with the
        boxes drawn.

    Example:
        The box ``[0.25, 0.25, 0.5, 0.5]`` on an image of width ``8``
        and height ``4`` becomes the pixel box ``[2, 1, 6, 3]``:

        >>> import torch
        >>> img = torch.zeros(3, 4, 8, dtype=torch.uint8)
        >>> label = torch.tensor([[0.25, 0.25, 0.5, 0.5]])
        >>> draw_bounding_box_labels(img, label, colors="red")[0].tolist()
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 255, 255, 255, 255, 255, 0],
         [0, 0, 255, 0, 0, 0, 255, 0],
         [0, 0, 255, 255, 255, 255, 255, 0]]

    """
    _, H, W = img.shape
    bboxs = box_convert(label, "xywh", "xyxy")
    bboxs[:, 0::2] *= W
    bboxs[:, 1::2] *= H
    return draw_bounding_boxes(img, bboxs, **kwargs)


def draw_keypoint_labels(img: Tensor, label: Tensor, **kwargs) -> Tensor:
    """Draw normalized keypoints on an image.

    The function scales the ``x`` and ``y`` values by the image width
    and height, truncates them to integers, and calls
    ``torchvision.utils.draw_keypoints``. It draws every keypoint and
    ignores the visibility.

    **Side effect:** when ``label`` is contiguous, the function scales
    the ``x`` and ``y`` values of ``label`` in place.

    Args:
        img (``Tensor``): A ``uint8`` image of shape ``[C, H, W]``.
        label (``Tensor``): Keypoints of shape ``[N, 3K]``, with rows
            ``[x_1, y_1, v_1, ..., x_K, y_K, v_K]``. The coordinates are
            normalized to ``[0, 1]``.
        **kwargs (``Any``): Keyword arguments forwarded to
            ``draw_keypoints``, such as ``colors``, ``radius``, and
            ``connectivity``.

    Returns:
        ``Tensor``: A new image of the same shape as ``img``, with the
        keypoints drawn. ``img`` itself when ``label`` holds no
        keypoints.

    Example:
        The keypoint lands on the pixel in row ``1`` and column ``2``.
        The call also changes ``label`` to pixel coordinates:

        >>> import torch
        >>> img = torch.zeros(3, 4, 4, dtype=torch.uint8)
        >>> label = torch.tensor([[0.5, 0.25, 2.0]])
        >>> out = draw_keypoint_labels(img, label, colors="red", radius=1)
        >>> out[0, 1, 2].item(), label.tolist()
        (255, [[2.0, 1.0, 2.0]])

    """
    _, H, W = img.shape
    keypoints_unflat = label.reshape(-1, 3)
    keypoints_points = keypoints_unflat[:, :2]
    keypoints_points[:, 0] *= W
    keypoints_points[:, 1] *= H

    n_instances = label.shape[0]
    if n_instances == 0:
        out_keypoints = keypoints_points.reshape((-1, 2)).unsqueeze(0).int()
    else:
        out_keypoints = keypoints_points.reshape((n_instances, -1, 2)).int()

    if out_keypoints.numel() == 0:
        return img
    return draw_keypoints(img, out_keypoints, **kwargs)


def denormalize(
    img: Tensor,
    mean: list[float] | float | None = None,
    std: list[float] | float | None = None,
    to_uint8: bool = False,
) -> Tensor:
    r"""Undo the normalization of an image.

    For each channel :math:`c`, the function computes

    .. math::

        y_c = x_c \sigma_c + \mu_c

    where :math:`\mu` is ``mean`` and :math:`\sigma` is ``std``. With
    ``to_uint8``, it then multiplies the result by ``255``, clips it to
    ``[0, 255]``, and truncates it to ``uint8``.

    Args:
        img (``Tensor``): A normalized image of shape ``[C, H, W]``.
        mean (list[float] | float | None): The mean of the normalization,
            one value for each channel or one ``float`` for all channels.
            ``None`` selects ``0``.
        std (list[float] | float | None): The standard deviation of the
            normalization, in the same form as ``mean``. ``None`` selects
            ``1``.
        to_uint8 (bool): Whether to scale the result to a ``uint8``
            image.

    Returns:
        ``Tensor``: A new image of shape ``[C, H, W]``, of dtype ``uint8``
        with ``to_uint8``.

    Example:
        >>> import torch
        >>> img = torch.tensor([[[0.0]], [[1.0]], [[-1.0]]])
        >>> denormalize(img, mean=0.5, std=0.25).flatten().tolist()
        [0.5, 0.75, 0.25]
        >>> image = denormalize(img, mean=0.5, std=0.25, to_uint8=True)
        >>> image.flatten().tolist()
        [127, 191, 63]

    """
    mean = mean or 0
    std = std or 1
    if isinstance(mean, float):
        mean = [mean] * img.shape[0]
    if isinstance(std, float):
        std = [std] * img.shape[0]
    mean_tensor = torch.tensor(mean, device=img.device)
    std_tensor = torch.tensor(std, device=img.device)
    new_mean = -mean_tensor / std_tensor
    new_std = 1 / std_tensor
    out_img = TF.normalize(img, mean=new_mean.tolist(), std=new_std.tolist())
    if to_uint8:
        out_img = out_img.mul_(255).clamp_(0, 255).to(torch.uint8)
    return out_img


# TODO: This should be left to the loader
def get_denormalized_images(cfg: Config, images: Tensor) -> Tensor:
    """Convert a batch of model input images to ``uint8`` images.

    When ``trainer.preprocessing.normalize`` is active, the function
    reads ``mean`` and ``std`` from its ``params``, and `preprocess_images`
    undoes the normalization. A missing key selects the ImageNet value,
    ``[0.485, 0.456, 0.406]`` for ``mean`` and ``[0.229, 0.224, 0.225]``
    for ``std``. When the normalization is not active, the function only
    casts the images to ``uint8``.

    `LuxonisLightningModule` and the inference utilities use the result
    as the canvas of the visualizers. `GradCamCallback` draws its heat
    maps on it.

    Args:
        cfg (Config): The config of the model.
        images (``Tensor``): The input images of shape ``[B, C, H, W]``,
            as the loader returns them.

    Returns:
        ``Tensor``: A new ``uint8`` tensor of shape ``[B, C, H, W]``.

    """
    normalize_params = cfg.trainer.preprocessing.normalize.params
    mean = std = None
    if cfg.trainer.preprocessing.normalize.active:
        mean = normalize_params.get("mean", [0.485, 0.456, 0.406])
        std = normalize_params.get("std", [0.229, 0.224, 0.225])
    return preprocess_images(images, mean=mean, std=std)  # type: ignore


def number_to_hsl(seed: int) -> tuple[float, float, float]:
    """Map an integer to a hue, with fixed saturation and lightness.

    The hue is ``(seed * 157) % 360``. The prime factor spreads the hues
    of consecutive seeds over the color wheel. The saturation is ``0.8``
    and the lightness is ``0.5``.

    Args:
        seed (int): The integer to map.

    Returns:
        tuple[float, float, float]: The hue in degrees, in ``[0, 360)``,
        the saturation, and the lightness.

    Example:
        >>> number_to_hsl(3)
        (111, 0.8, 0.5)

    """
    # Use a prime number to spread the hues more evenly
    # and ensure they are visually distinguishable
    hue = (seed * 157) % 360
    saturation = 0.8  # Fixed saturation
    lightness = 0.5  # Fixed lightness
    return (hue, saturation, lightness)


def hsl_to_rgb(hsl: tuple[float, float, float]) -> Color:
    """Convert an HSL color to an 8-bit RGB color.

    The function truncates each channel to an integer.

    Args:
        hsl (tuple[float, float, float]): The hue in degrees, in
            ``[0, 360)``, and the saturation and the lightness, in
            ``[0, 1]``.

    Returns:
        tuple[int, int, int]: The red, green, and blue values, in
        ``[0, 255]``.

    Example:
        >>> hsl_to_rgb((0, 1.0, 0.5))
        (255, 0, 0)

    """
    r, g, b = colorsys.hls_to_rgb(hsl[0] / 360, hsl[2], hsl[1])
    return int(r * 255), int(g * 255), int(b * 255)


def get_color(seed: int) -> Color:
    """Return a distinct RGB color for an integer.

    The color is not random. The same ``seed`` always gives the same
    color. The function adds ``45`` to ``seed`` and converts the result
    with `number_to_hsl` and `hsl_to_rgb`. The visualizers pass the
    class index as ``seed``.

    Args:
        seed (int): The integer that selects the color.

    Returns:
        tuple[int, int, int]: The red, green, and blue values, in
        ``[0, 255]``.

    Example:
        >>> get_color(0)
        (25, 76, 229)

    """
    return hsl_to_rgb(number_to_hsl(seed + 45))


def dynamically_determine_font_scale(
    height: int,
    width: int,
    thickness: int,
    font_scale: float | None = None,
    scale_factor: float = 500.0,
) -> tuple[float, int]:
    r"""Select a font scale and a line thickness for an image size.

    Without ``font_scale``, the function derives the scale from a
    weighted mean of the height :math:`H` and the width :math:`W`:

    .. math::

        w = \min\left(0.4, \frac{W}{10 \max(H, 1)}\right), \quad
        s = \frac{(1 - w) H + w W}{\text{scale\_factor}}

    A scale below ``1`` gets the thickness ``1``. A scale of ``1`` or
    more keeps ``thickness``.

    Args:
        height (int): The image height, in pixels.
        width (int): The image width, in pixels.
        thickness (int): The line thickness for a scale of ``1`` or
            more.
        font_scale (float | None): A fixed font scale. ``None`` selects
            the computed scale.
        scale_factor (float): The effective image size, in pixels, that
            gives the scale ``1``.

    Returns:
        tuple[float, int]: The font scale and the line thickness.

    Example:
        >>> dynamically_determine_font_scale(500, 500, thickness=2)
        (1.0, 2)
        >>> dynamically_determine_font_scale(100, 200, thickness=2)
        (0.24, 1)

    """
    aspect_ratio = width / max(height, 1)
    width_weight = min(0.4, aspect_ratio / 10.0)
    effective_size = height * (1 - width_weight) + width * width_weight

    computed_font_scale = (
        font_scale if font_scale is not None else effective_size / scale_factor
    )

    if computed_font_scale < 1:
        return computed_font_scale, 1
    return computed_font_scale, thickness


def potentially_upscale_masks(
    image_masks: Tensor, scale: float = 1.0
) -> Tensor:
    """Resize segmentation masks by a factor with nearest interpolation.

    The new height and width are ``int(H * scale)`` and
    ``int(W * scale)``, so a factor below ``1`` shrinks the masks. With
    a factor of ``1``, the function returns ``image_masks`` unchanged.

    Args:
        image_masks (``Tensor``): Masks of shape ``[N, H, W]``. Every
            non-zero value marks a pixel of a mask.
        scale (float): The resize factor.

    Returns:
        ``Tensor``: Boolean masks of shape
        ``[N, int(H * scale), int(W * scale)]``, or ``image_masks``
        itself when ``scale`` is ``1``.

    Example:
        >>> import torch
        >>> masks = torch.tensor([[[True, False], [False, True]]])
        >>> potentially_upscale_masks(masks, scale=2.0).int().tolist()
        [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]]

    """
    if scale is not None and scale != 1:
        image_masks = image_masks.unsqueeze(1)
        H_orig, W_orig = image_masks.shape[-2:]
        H_up = int(H_orig * scale)
        W_up = int(W_orig * scale)

        image_masks = F.interpolate(
            image_masks.float(), size=(H_up, W_up), mode="nearest"
        ).bool()
        return image_masks.squeeze(1).bool()
    return image_masks


# TODO: Support native visualizations
# NOTE: Ignore for now, native visualizations not a priority.
#
# It could be beneficial in the long term to make the visualization more abstract.
# Reason for that is that certain services, e.g. WandB, have their native way
# of visualizing things. So by restricting ourselves to only produce bitmap images
# for logging, we are limiting ourselves in how we can utilize those services.
# (I know we want to leave WandB and I don't know whether mlcloud offers anything
# similar, but it might save us some time in the future).')
#
# The idea would be that every visualizer would not only produce the bitmap
# images, but also some standardized representation of the visualizations.
# This would be sent to the logger, which would then decide how to log it.
# By default, it would log it as a bitmap image, but if we know we are logging
# to (e.g.) WandB, we could use the native WandB visualizations.
# Since we already have to check what logging is being used (to call the correct
# service), it should be somehow easy to implement.
#
# The more specific implementation/protocol could be, that every instance
# of `LuxonisVisualizer` would produce a tuple of
# (bitmap_visualizations, structured_visualizations).
#
# The `bitmap_visualizations` would be one of the following:
# - a single tensor (e.g. image)
#   - in this case, the tensor would be logged as a bitmap image
# - a tuple of two tensors
#   - in this case, the first tensor is considered labels and the second predictions
#   - e.g. GT and predicted segmentation mask
# - a tuple of a tensor and a list of tensors
#   - in this case, the first is considered labels
#     and the second unrelated predictions
# - an iterable of tensors
#   - in this case, the tensors are considered unrelated predictions
#
# The `structured_visualizations` would be have similar format, but  instead of
# tensors, it would consist of some structured data (e.g. dict of lists or something).
# We could even create a validation schema for this to enforce the structure.
# We would then just have to support this new structure in the logger (`LuxonisTracker`).
#
#  TEST:
def combine_visualizations(
    visualization: Tensor
    | tuple[Tensor, Tensor]
    | tuple[Tensor, list[Tensor]],
) -> Tensor:
    """Combine the output of a visualizer into one image batch.

    The trainer calls this function on the result of each
    `BaseVisualizer.run`:

    - A single tensor: the function returns it unchanged.
    - A pair ``(labels, predictions)`` of tensors, in a tuple or a list:
      the function resizes both batches to the larger height. Each batch
      keeps its aspect ratio. The function then puts the batches side by
      side, with the labels on the left.
    - A tensor and a list or a tuple of tensors: the function raises
      ``NotImplementedError``.

    Args:
        visualization (``Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]``):
            The output of a visualizer. The images have the shape
            ``[B, C, H, W]``, and the two batches of a pair have the same
            ``B`` and ``C``.

    Returns:
        ``Tensor``: The images of shape ``[B, C, H, W]``. For a pair,
        ``H`` is the larger height and ``W`` is the sum of the resized
        widths.

    Raises:
        NotImplementedError: When the second item is a list or a tuple
            of tensors.
        ValueError: When ``visualization`` has any other form.

    Example:
        >>> import torch
        >>> labels = torch.zeros(1, 3, 4, 4, dtype=torch.uint8)
        >>> predictions = torch.zeros(1, 3, 8, 6, dtype=torch.uint8)
        >>> combine_visualizations((labels, predictions)).shape
        torch.Size([1, 3, 8, 14])
        >>> combine_visualizations(labels) is labels
        True

    """
    match visualization:
        case Tensor() as viz:
            return viz
        case (Tensor(data=viz_labels), Tensor(data=viz_predictions)):
            viz_labels, viz_predictions = _resize_to_match(
                viz_labels, viz_predictions
            )
            return torch.cat([viz_labels, viz_predictions], dim=-1)

        case (Tensor(data=_), [*viz]) if isinstance(viz, list) and all(
            isinstance(v, Tensor) for v in viz
        ):
            raise NotImplementedError(
                "Composition of multiple visualizations not yet supported."
            )
        case _:
            raise ValueError(
                "Visualization should be either a single tensor or a tuple of "
                "two tensors or a tuple of a tensor and a list of tensors. "
                f"Got: `{type(visualization)}`."
            )


def _target_size_for_keep_size(
    keep_size: Literal["larger", "smaller", "first", "second"],
    w1: int,
    h1: int,
    w2: int,
    h2: int,
) -> tuple[int, int]:
    if keep_size == "larger":
        return max(w1, w2), max(h1, h2)
    if keep_size == "smaller":
        return min(w1, w2), min(h1, h2)
    if keep_size == "first":
        return w1, h1
    if keep_size == "second":
        return w2, h2
    raise ValueError(
        f"Invalid value for keep_size: {keep_size}. "
        "Valid options are: 'larger', 'smaller', 'first', 'second'."
    )


def _fit_to_aspect_ratio(
    target_width: int,
    target_height: int,
    aspect_ratio: float,
    resize_along: Literal["width", "height", "exact"],
) -> tuple[int, int]:
    if resize_along == "width" or (
        resize_along == "exact" and target_width / target_height > aspect_ratio
    ):
        return target_width, int(target_width / aspect_ratio)
    return int(target_height * aspect_ratio), target_height


def _resize_to_match(
    fst: Tensor,
    snd: Tensor,
    *,
    keep_size: Literal["larger", "smaller", "first", "second"] = "larger",
    resize_along: Literal["width", "height", "exact"] = "height",
    keep_aspect_ratio: bool = True,
) -> tuple[Tensor, Tensor]:
    """Resize two images, so that they can be concatenated.

    ``keep_size`` selects the target width and height:

    - ``"larger"``: the larger width and the larger height.
    - ``"smaller"``: the smaller width and the smaller height.
    - ``"first"``: the size of ``fst``.
    - ``"second"``: the size of ``snd``.

    ``resize_along`` selects the dimension that both images share:

    - ``"height"``: the target height. The target width becomes the
      width of ``fst`` for ``"larger"`` and ``"first"``, else the width
      of ``snd``.
    - ``"width"``: the target width. The target height becomes the
      height of ``fst`` for ``"larger"`` and ``"first"``, else the
      height of ``snd``.
    - ``"exact"``: the target width and the target height.

    With ``keep_aspect_ratio``, each image keeps its aspect ratio, and
    the function computes its other dimension, truncated to an integer.
    For ``"exact"``, an image matches the target width when the target
    has a larger aspect ratio than the image. Otherwise the image matches
    the target height. Thus the other dimension can exceed the target.
    Without ``keep_aspect_ratio``, both images get the target width and
    height.

    Args:
        fst (``Tensor``): The first image, of shape ``[..., H, W]``.
        snd (``Tensor``): The second image, of shape ``[..., H, W]``.
        keep_size (``Literal["larger", "smaller", "first", "second"]``):
            The rule for the target size.
        resize_along (``Literal["width", "height", "exact"]``): The
            dimension that the images share.
        keep_aspect_ratio (bool): Whether each image keeps its aspect
            ratio.

    Returns:
        ``tuple[Tensor, Tensor]``: The resized ``fst`` and ``snd``.

    Raises:
        ValueError: When ``resize_along`` or ``keep_size`` is not one of
            the values above.

    """
    if resize_along not in ["width", "height", "exact"]:
        raise ValueError(
            f"Invalid value for resize_along: {resize_along}. "
            "Valid options are: 'width', 'height', 'exact'."
        )

    *_, h1, w1 = fst.shape
    *_, h2, w2 = snd.shape

    target_width, target_height = _target_size_for_keep_size(
        keep_size, w1, h1, w2, h2
    )

    if resize_along == "width":
        target_height = h1 if keep_size in ["first", "larger"] else h2
    elif resize_along == "height":
        target_width = w1 if keep_size in ["first", "larger"] else w2

    if keep_aspect_ratio:
        target_width_fst, target_height_fst = _fit_to_aspect_ratio(
            target_width, target_height, w1 / h1, resize_along
        )
        target_width_snd, target_height_snd = _fit_to_aspect_ratio(
            target_width, target_height, w2 / h2, resize_along
        )
    else:
        target_width_fst, target_height_fst = target_width, target_height
        target_width_snd, target_height_snd = target_width, target_height

    fst_resized = TF.resize(fst, [target_height_fst, target_width_fst])
    snd_resized = TF.resize(snd, [target_height_snd, target_width_snd])

    return fst_resized, snd_resized
