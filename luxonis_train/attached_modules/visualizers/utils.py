import colorsys
import io
from typing import Literal

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


def figure_to_torch(fig: Figure, width: int, height: int) -> Tensor:
    """Converts a matplotlib `Figure` to a `Tensor`."""
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
    """Converts a torch image (CHW) to a numpy array (HWC).

    @type img: Tensor
    @param img: Torch image (CHW)
    @type reverse_colors: bool
    @param reverse_colors: Whether to reverse colors (RGB to BGR).
        Defaults to False.
    @rtype: npt.NDArray[np.uint8]
    @return: Numpy image (HWC)
    """
    if img.is_floating_point():
        img = img.mul(255).int()
    img = torch.clamp(img, 0, 255)
    arr = img.detach().cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
    arr = np.ascontiguousarray(arr)
    if reverse_colors:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return arr


def numpy_to_torch_img(img: np.ndarray) -> Tensor:
    """Converts numpy image (HWC) to torch image (CHW)."""
    return torch.from_numpy(img).permute(2, 0, 1)


def preprocess_images(
    imgs: Tensor,
    mean: list[float] | float | None = None,
    std: list[float] | float | None = None,
) -> Tensor:
    """Performs preprocessing on a batch of images.

    Preprocessing includes denormalizing and converting to uint8.

    @type imgs: Tensor
    @param imgs: Batch of images.
    @type mean: list[float] | float | None
    @param mean: Mean used for denormalization. Defaults to C{None}.
    @type std: list[float] | float | None
    @param std: Std used for denormalization. Defaults to C{None}.
    @rtype: Tensor
    @return: Batch of preprocessed images.
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
    """Draws segmentation labels on an image.

    @type image: Tensor
    @param image: Image to draw on.
    @type target: Tensor
    @param target: Segmentation label.
    @type alpha: float
    @param alpha: Alpha value for blending. Defaults to C{0.4}.
    @rtype: Tensor
    @return: Image with segmentation labels drawn on.
    """
    masks = target.bool()
    masks = masks.cpu()
    image = image.cpu()
    return draw_segmentation_masks(image, masks, alpha=alpha, colors=colors)


def draw_bounding_box_labels(img: Tensor, label: Tensor, **kwargs) -> Tensor:
    """Draws bounding box labels on an image.

    @type img: Tensor
    @param img: Image to draw on.
    @type label: Tensor
    @param label: Bounding box label. The shape should be (n_instances,
        4), where the last dimension is (x, y, w, h).
    @type kwargs: dict
    @param kwargs: Additional arguments to pass to
        L{torchvision.utils.draw_bounding_boxes}.
    @rtype: Tensor
    @return: Image with bounding box labels drawn on.
    """
    _, H, W = img.shape
    bboxs = box_convert(label, "xywh", "xyxy")
    bboxs[:, 0::2] *= W
    bboxs[:, 1::2] *= H
    return draw_bounding_boxes(img, bboxs, **kwargs)


def draw_keypoint_labels(img: Tensor, label: Tensor, **kwargs) -> Tensor:
    """Draws keypoint labels on an image.

    @type img: Tensor
    @param img: Image to draw on.
    @type label: Tensor
    @param label: Keypoint label. The shape should be (n_instances, 3),
        where the last dimension is (x, y, visibility).
    @type kwargs: dict
    @param kwargs: Additional arguments to pass to
        L{torchvision.utils.draw_keypoints}.
    @rtype: Tensor
    @return: Image with keypoint labels drawn on.
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
    """Denormalizes an image back to original values, optionally
    converts it to uint8.

    @type img: Tensor
    @param img: Image to denormalize.
    @type mean: list[float] | float | None
    @param mean: Mean used for denormalization. Defaults to C{None}.
    @type std: list[float] | float | None
    @param std: Std used for denormalization. Defaults to C{None}.
    @type to_uint8: bool
    @param to_uint8: Whether to convert to uint8. Defaults to C{False}.
    @rtype: Tensor
    @return: denormalized image.
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
    normalize_params = cfg.trainer.preprocessing.normalize.params
    mean = std = None
    if cfg.trainer.preprocessing.normalize.active:
        mean = normalize_params.get("mean", [0.485, 0.456, 0.406])
        std = normalize_params.get("std", [0.229, 0.224, 0.225])
    return preprocess_images(images, mean=mean, std=std)  # type: ignore


def number_to_hsl(seed: int) -> tuple[float, float, float]:
    """Map a number to a distinct HSL color."""
    # Use a prime number to spread the hues more evenly
    # and ensure they are visually distinguishable
    hue = (seed * 157) % 360
    saturation = 0.8  # Fixed saturation
    lightness = 0.5  # Fixed lightness
    return (hue, saturation, lightness)


def hsl_to_rgb(hsl: tuple[float, float, float]) -> Color:
    """Convert HSL color to RGB."""
    r, g, b = colorsys.hls_to_rgb(hsl[0] / 360, hsl[2], hsl[1])
    return int(r * 255), int(g * 255), int(b * 255)


def get_color(seed: int) -> Color:
    """Generates a random color from a seed.

    @type seed: int
    @param seed: Seed to use for the generator.
    @rtype: L{Color}
    @return: Generated color.
    """
    return hsl_to_rgb(number_to_hsl(seed + 45))


def dynamically_determine_font_scale(
    height: int,
    width: int,
    thickness: int,
    font_scale: float | None = None,
    scale_factor: float = 500.0,
) -> tuple[float, int]:
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
    """Upscales boolean segmentation masks.

    @param image_masks:
    @param scale: scale factor
    @return: Upscaled image masks
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
    """Default way of combining multiple visualizations into one final
    image."""

    def resize_to_match(
        fst: Tensor,
        snd: Tensor,
        *,
        keep_size: Literal["larger", "smaller", "first", "second"] = "larger",
        resize_along: Literal["width", "height", "exact"] = "height",
        keep_aspect_ratio: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Resizes two images so they have the same size.

        Resizes two images so they can be concateneted together. It's possible to
        configure how the images are resized.

        @type fst: Tensor[C, H, W]
        @param fst: First image.
        @type snd: Tensor[C, H, W]
        @param snd: Second image.
        @type keep_size: Literal["larger", "smaller", "first", "second"]
        @param keep_size: Which size to keep. Options are:
            - "larger": Resize the smaller image to match the size of the larger image.
            - "smaller": Resize the larger image to match the size of the smaller image.
            - "first": Resize the second image to match the size of the first image.
            - "second": Resize the first image to match the size of the second image.

        @type resize_along: Literal["width", "height", "exact"]
        @param resize_along: Which dimensions to match. Options are:
            - "width": Resize images along the width dimension.
            - "height": Resize images along the height dimension.
            - "exact": Resize images to match both width and height dimensions.

        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep the aspect ratio of the images.
            Only takes effect when the "exact" option is selected for the
            C{resize_along} argument. Defaults to C{True}.

        @rtype: tuple[Tensor[C, H, W], Tensor[C, H, W]]
        @return: Resized images.
        """
        if resize_along not in ["width", "height", "exact"]:
            raise ValueError(
                "Invalid value for resize_along: {resize_along}. "
                "Valid options are: 'width', 'height', 'exact'."
            )

        *_, h1, w1 = fst.shape

        *_, h2, w2 = snd.shape

        if keep_size == "larger":
            target_width = max(w1, w2)
            target_height = max(h1, h2)
        elif keep_size == "smaller":
            target_width = min(w1, w2)
            target_height = min(h1, h2)
        elif keep_size == "first":
            target_width = w1
            target_height = h1
        elif keep_size == "second":
            target_width = w2
            target_height = h2
        else:
            raise ValueError(
                f"Invalid value for keep_size: {keep_size}. "
                "Valid options are: 'larger', 'smaller', 'first', 'second'."
            )

        if resize_along == "width":
            target_height = h1 if keep_size in ["first", "larger"] else h2
        elif resize_along == "height":
            target_width = w1 if keep_size in ["first", "larger"] else w2

        if keep_aspect_ratio:
            ar1 = w1 / h1
            ar2 = w2 / h2
            if resize_along == "width" or (
                resize_along == "exact" and target_width / target_height > ar1
            ):
                target_height_fst = int(target_width / ar1)
                target_width_fst = target_width
            else:
                target_width_fst = int(target_height * ar1)
                target_height_fst = target_height
            if resize_along == "width" or (
                resize_along == "exact" and target_width / target_height > ar2
            ):
                target_height_snd = int(target_width / ar2)
                target_width_snd = target_width
            else:
                target_width_snd = int(target_height * ar2)
                target_height_snd = target_height
        else:
            target_width_fst, target_height_fst = target_width, target_height
            target_width_snd, target_height_snd = target_width, target_height

        fst_resized = TF.resize(fst, [target_height_fst, target_width_fst])
        snd_resized = TF.resize(snd, [target_height_snd, target_width_snd])

        return fst_resized, snd_resized

    match visualization:
        case Tensor() as viz:
            return viz
        case (Tensor(data=viz_labels), Tensor(data=viz_predictions)):
            viz_labels, viz_predictions = resize_to_match(
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
