import colorsys
import io
import warnings
from typing import List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torchvision.ops import box_convert
from torchvision.utils import (
    _log_api_usage_once,
    _parse_colors,
    draw_bounding_boxes,
    draw_keypoints,
    draw_segmentation_masks,
)

from luxonis_train.utils import Config, xywhr2xyxyxyxy, xyxyxyxy2xywhr

Color = str | tuple[int, int, int]
"""Color type alias.

Can be either a string (e.g. "red", "#FF5512") or a tuple of RGB values.
"""


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
    """Converts a torch image (CHW) to a numpy array (HWC). Optionally
    also converts colors.

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

    Preprocessing includes unnormalizing and converting to uint8.

    @type imgs: Tensor
    @param imgs: Batch of images.
    @type mean: list[float] | float | None
    @param mean: Mean used for unnormalization. Defaults to C{None}.
    @type std: list[float] | float | None
    @param std: Std used for unnormalization. Defaults to C{None}.
    @rtype: Tensor
    @return: Batch of preprocessed images.
    """
    out_imgs = []
    for i in range(imgs.shape[0]):
        curr_img = imgs[i]
        if mean is not None or std is not None:
            curr_img = unnormalize(curr_img, to_uint8=True, mean=mean, std=std)
        else:
            curr_img = curr_img.to(torch.uint8)

        out_imgs.append(curr_img)

    return torch.stack(out_imgs)


def draw_segmentation_labels(
    img: Tensor,
    label: Tensor,
    alpha: float = 0.4,
    colors: Color | list[Color] | None = None,
) -> Tensor:
    """Draws segmentation labels on an image.

    @type img: Tensor
    @param img: Image to draw on.
    @type label: Tensor
    @param label: Segmentation label.
    @type alpha: float
    @param alpha: Alpha value for blending. Defaults to C{0.4}.
    @rtype: Tensor
    @return: Image with segmentation labels drawn on.
    """
    masks = label.bool()
    masks = masks.cpu()
    img = img.cpu()
    return draw_segmentation_masks(img, masks, alpha=alpha, colors=colors)


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


def draw_obounding_box(
    img: Tensor, obbox: Tensor | np.ndarray, **kwargs
) -> Tensor:
    """Draws oriented bounding box (obb) labels on an image.

    @type img: Tensor
    @param img: Image to draw on.
    @type obbox: Tensor
    @param obbox: Oriented bounding box. The shape should be
        (n_instances, 8) or (n_instances, 5), where the last dimension
        is (x1, y1, x2, y2, x3, y3, x4, y4) or (xc, yc, w, h, r).
    @type kwargs: dict
    @param kwargs: Additional arguments to pass to
        L{draw_obounding_boxes}.
    @rtype: Tensor
    @return: Image with bounding box labels drawn on.
    """
    _, H, W = img.shape
    # The conversion below is needed for fitting a rectangle to the 4 label points, which can form
    # a polygon sometimes
    if obbox.shape[-1] > 5:
        obbox = xyxyxyxy2xywhr(obbox)  # xywhr
    bboxs_2 = xywhr2xyxyxyxy(obbox)  # shape: (bs, 4, 2)
    if isinstance(bboxs_2, np.ndarray):
        bboxs_2 = torch.tensor(bboxs_2)
    if bboxs_2.numel() == 0:
        raise ValueError
    bboxs = bboxs_2.view(bboxs_2.size(0), -1)  # x1y1x2y2x3y3x4y4
    bboxs[:, 0::2] *= W
    bboxs[:, 1::2] *= H
    return draw_obounding_boxes(img, bboxs, **kwargs)


def draw_obounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[
        Union[
            List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]
        ]
    ] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:
    """Draws oriented bounding boxes (obb) on given RGB image. The image
    values should be uint8 in [0, 255] or float in [0, 1]. If fill is
    True, Resulting Tensor should be saved as PNG image.

    Args:
        image (Tensor): Tensor of shape (C, H, W) and dtype uint8 or float.
        boxes (Tensor): Tensor of size (N, 8) containing bounding boxes in (x1, y1, x2, y2, x3, y3, x4, y4)
            format. Note that the boxes are absolute coordinates with respect to the image. In other words: `0 <= x < W` and
            `0 <= y < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.

    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """
    import torchvision.transforms.v2.functional as F  # noqa

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(draw_obounding_boxes)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif not (image.dtype == torch.uint8 or image.is_floating_point()):
        raise ValueError(
            f"The image dtype must be uint8 or float, got {image.dtype}"
        )
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
    # elif (boxes[:, 0] > boxes[:, 2]).any() or (boxes[:, 1] > boxes[:, 3]).any():
    #     raise ValueError(
    #         "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
    #     )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    colors = _parse_colors(colors, num_objects=num_boxes)

    if font is None:
        if font_size is not None:
            warnings.warn(
                "Argument 'font_size' will be ignored since 'font' is not set."
            )
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    original_dtype = image.dtype
    if original_dtype.is_floating_point:
        image = F.to_dtype(image, dtype=torch.uint8, scale=True)

    img_to_draw = F.to_pil_image(image)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.polygon(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.polygon(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text(
                (bbox[0] + margin, bbox[1] + margin),
                label,
                fill=color,
                font=txt_font,
            )

    out = F.pil_to_tensor(img_to_draw)
    if original_dtype.is_floating_point:
        out = F.to_dtype(out, dtype=original_dtype, scale=True)
    return out


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
    keypoints_unflat = label[:, 1:].reshape(-1, 3)
    keypoints_points = keypoints_unflat[:, :2]
    keypoints_points[:, 0] *= W
    keypoints_points[:, 1] *= H

    n_instances = label.shape[0]
    if n_instances == 0:
        out_keypoints = keypoints_points.reshape((-1, 2)).unsqueeze(0).int()
    else:
        out_keypoints = keypoints_points.reshape((n_instances, -1, 2)).int()

    return draw_keypoints(img, out_keypoints, **kwargs)


def seg_output_to_bool(data: Tensor, binary_threshold: float = 0.5) -> Tensor:
    """Converts seg head output to 2D boolean mask for visualization."""
    masks = torch.empty_like(data, dtype=torch.bool, device=data.device)
    if data.shape[0] == 1:
        classes = torch.sigmoid(data)
        masks[0] = classes >= binary_threshold
    else:
        classes = torch.argmax(data, dim=0)
        for i in range(masks.shape[0]):
            masks[i] = classes == i
    return masks


def unnormalize(
    img: Tensor,
    mean: list[float] | float | None = None,
    std: list[float] | float | None = None,
    to_uint8: bool = False,
) -> Tensor:
    """Unnormalizes an image back to original values, optionally
    converts it to uint8.

    @type img: Tensor
    @param img: Image to unnormalize.
    @type mean: list[float] | float | None
    @param mean: Mean used for unnormalization. Defaults to C{None}.
    @type std: list[float] | float | None
    @param std: Std used for unnormalization. Defaults to C{None}.
    @type to_uint8: bool
    @param to_uint8: Whether to convert to uint8. Defaults to C{False}.
    @rtype: Tensor
    @return: Unnormalized image.
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
    out_img = F.normalize(img, mean=new_mean.tolist(), std=new_std.tolist())
    if to_uint8:
        out_img = torch.clamp(out_img.mul(255), 0, 255).to(torch.uint8)
    return out_img


def get_unnormalized_images(cfg: Config, inputs: dict[str, Tensor]) -> Tensor:
    # Get images from inputs according to config
    images = inputs[cfg.loader.image_source]

    normalize_params = cfg.trainer.preprocessing.normalize.params
    mean = std = None
    if cfg.trainer.preprocessing.normalize.active:
        mean = normalize_params.get("mean", [0.485, 0.456, 0.406])
        std = normalize_params.get("std", [0.229, 0.224, 0.225])
    return preprocess_images(
        images,
        mean=mean,
        std=std,
    )


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
                "two tensors or a tuple of a tensor and a list of tensors."
                f"Got: `{type(visualization)}`."
            )
