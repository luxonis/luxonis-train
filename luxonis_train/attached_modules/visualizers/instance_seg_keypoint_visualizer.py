from torch import Tensor

from luxonis_train.attached_modules.visualizers.base_visualizer import (
    BaseVisualizer,
)
from luxonis_train.attached_modules.visualizers.instance_segmentation_visualizer import (
    InstanceSegmentationVisualizer,
)
from luxonis_train.attached_modules.visualizers.keypoint_visualizer import (
    KeypointVisualizer,
)
from luxonis_train.attached_modules.visualizers.utils import Color, get_color
from luxonis_train.tasks import Tasks


class InstanceSegKeypointVisualizer(BaseVisualizer):
    supported_tasks = [Tasks.INSTANCE_SEGMENTATION_KEYPOINTS]

    def __init__(
        self,
        labels: dict[int, str] | list[str] | None = None,
        draw_labels: bool = True,
        draw_scores: bool = False,
        colors: dict[str, Color] | list[Color] | None = None,
        fill: bool = False,
        width: int | None = None,
        font: str | None = None,
        font_size: int | None = None,
        alpha: float = 0.6,
        visibility_threshold: float = 0.5,
        connectivity: list[tuple[int, int]] | None = None,
        visible_color: Color = "red",
        nonvisible_color: Color | None = None,
        radius: int | None = None,
        draw_indices: bool = False,
        **kwargs,
    ):
        """
        @type labels: dict[int, str] | list[str] | None
        @param labels: Dictionary mapping class indices to class labels.
        @type draw_labels: bool
        @param draw_labels: Whether to draw class labels.
        @type draw_scores: bool
        @param draw_scores: Whether to append prediction confidence
            scores to the rendered labels. Defaults to C{False}.
        @type colors: dict[str, L{Color}] | list[L{Color}] | None
        @param colors: Dictionary mapping class labels to colors.
        @type fill: bool
        @param fill: Whether to fill bounding boxes.
        @type width: int | None
        @param width: Width of the bounding box lines.
        @type font: str | None
        @param font: TrueType font filename.
        @type font_size: int | None
        @param font_size: Font size for labels.
        @type alpha: float
        @param alpha: Alpha value for segmentation masks.
        @type visibility_threshold: float
        @param visibility_threshold: Threshold for keypoint visibility.
        @type connectivity: list[tuple[int, int]] | None
        @param connectivity: Keypoint skeleton connections.
        @type visible_color: L{Color}
        @param visible_color: Color for visible keypoints.
        @type nonvisible_color: L{Color} | None
        @param nonvisible_color: Color for non-visible keypoints.
        @type radius: int | None
        @param radius: Keypoint radius.
        @type draw_indices: bool
        @param draw_indices: Whether to draw keypoint indices.
        """
        super().__init__(**kwargs)

        if isinstance(labels, list):
            labels = dict(enumerate(labels))

        self._bbox_labels = labels or self.classes.inverse

        if colors is None:
            colors = {
                label: get_color(i) for i, label in self._bbox_labels.items()
            }
        if isinstance(colors, list):
            colors = {
                self._bbox_labels[i]: color for i, color in enumerate(colors)
            }

        self._colors = colors
        self._fill = fill
        self._width = width
        self._font = font
        self._font_size = font_size
        self._draw_labels = draw_labels
        self._draw_scores = draw_scores
        self._alpha = alpha

        self._visibility_threshold = visibility_threshold
        self._connectivity = connectivity
        self._visible_color = visible_color
        self._nonvisible_color = nonvisible_color
        self._radius = radius
        self._draw_indices = draw_indices

    def forward(
        self,
        prediction_canvas: Tensor,
        target_canvas: Tensor,
        boundingbox: list[Tensor],
        instance_segmentation: list[Tensor],
        keypoints: list[Tensor],
        target_boundingbox: Tensor | None,
        target_instance_segmentation: Tensor | None,
        target_keypoints: Tensor | None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        # Draw predictions: masks + bboxes + keypoints
        pred_viz = InstanceSegmentationVisualizer.draw_predictions(
            prediction_canvas,
            boundingbox,
            instance_segmentation,
            self._width,
            self._bbox_labels,
            self._colors,
            self._draw_labels,
            self._draw_scores,
            self._alpha,
            self._scale,
        )

        prediction_radius = (
            KeypointVisualizer._get_radius(prediction_canvas)
            if self._radius is None
            else self._radius
        )

        pred_viz = KeypointVisualizer.draw_predictions(
            pred_viz,
            keypoints,
            self._draw_indices,
            connectivity=self._connectivity,
            nonvisible_color=self._nonvisible_color,
            visible_color=self._visible_color,
            visibility_threshold=self._visibility_threshold,
            radius=prediction_radius,
            scale=self._scale,
        )

        has_targets = (
            target_boundingbox is not None
            or target_instance_segmentation is not None
            or target_keypoints is not None
        )
        if not has_targets:
            return pred_viz

        target_viz = target_canvas

        if (
            target_boundingbox is not None
            and target_instance_segmentation is not None
        ):
            target_viz = InstanceSegmentationVisualizer.draw_targets(
                target_viz,
                target_boundingbox,
                target_instance_segmentation,
                self._width,
                self._bbox_labels,
                self._colors,
                self._draw_labels,
                self._alpha,
                self._scale,
            )

        if target_keypoints is not None:
            target_radius = (
                KeypointVisualizer._get_radius(target_canvas)
                if self._radius is None
                else self._radius
            )
            target_viz = KeypointVisualizer.draw_targets(
                target_viz,
                target_keypoints,
                self._draw_indices,
                radius=target_radius,
                colors=self._visible_color,
                connectivity=self._connectivity,
            )

        return target_viz, pred_viz
