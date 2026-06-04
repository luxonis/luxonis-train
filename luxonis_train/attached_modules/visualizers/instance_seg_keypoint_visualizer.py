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
    """Visualize instance segmentation and keypoint predictions.

    Metadata:
        - Module type: visualizer
        - Registry name: ``InstanceSegKeypointVisualizer``
        - Task: instance_segmentation_keypoints
        - Attached node types: None
        - Inputs: prediction and target canvases, ``boundingbox``,
          ``instance_segmentation``, and ``keypoints`` predictions, plus
          optional matching targets.
        - Outputs: prediction visualization, or ``(target_viz, pred_viz)``
          when any targets are provided.

    Provenance:
        - Source: Internal
        - License: Project license
        - Implementation notes: Composes the instance segmentation and
          keypoint visualizer drawing helpers.

    Prediction format:
        - ``boundingbox`` and ``instance_segmentation`` follow the instance
          segmentation visualizer formats.
        - ``keypoints`` is a list of per-image keypoint tensors with
          ``(x, y, visibility)`` values.

    Target format:
        - Target tensors mirror the bounding box, instance mask, and keypoint
          formats used by the composed visualizers.

    """

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
        """Initialize the instance segmentation keypoint visualizer.

        Args:
            labels (dict[int, str] | list[str] | None): Dictionary mapping class indices to class
                labels.
            draw_labels (bool): Whether to draw class labels.
            draw_scores (bool): Whether to append prediction confidence scores to the rendered
                labels. Defaults to ``False``.
            colors (dict[str, Color] | list[Color] | None): Dictionary mapping class labels to
                colors.
            fill (bool): Whether to fill bounding boxes.
            width (int | None): Width of the bounding box lines.
            font (str | None): TrueType font filename.
            font_size (int | None): Font size for labels.
            alpha (float): Alpha value for segmentation masks.
            visibility_threshold (float): Threshold for keypoint visibility.
            connectivity (list[tuple[int, int]] | None): Keypoint skeleton connections.
            visible_color (Color): Color for visible keypoints.
            nonvisible_color (Color | None): Color for non-visible keypoints.
            radius (int | None): Keypoint radius.
            draw_indices (bool): Whether to draw keypoint indices.
            **kwargs (Any): Keyword arguments forwarded to the parent class.

        """
        super().__init__(**kwargs)

        if isinstance(labels, list):
            labels = dict(enumerate(labels))

        self.bbox_labels = labels or self.classes.inverse

        if colors is None:
            colors = {
                label: get_color(i) for i, label in self.bbox_labels.items()
            }
        if isinstance(colors, list):
            colors = {
                self.bbox_labels[i]: color for i, color in enumerate(colors)
            }

        self.colors = colors
        self.fill = fill
        self.width = width
        self.font = font
        self.font_size = font_size
        self.draw_labels = draw_labels
        self.draw_scores = draw_scores
        self.alpha = alpha

        self.visibility_threshold = visibility_threshold
        self.connectivity = connectivity
        self.visible_color = visible_color
        self.nonvisible_color = nonvisible_color
        self.radius = radius
        self.draw_indices = draw_indices

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
            self.width,
            self.bbox_labels,
            self.colors,
            self.draw_labels,
            self.draw_scores,
            self.alpha,
            self.scale,
        )

        prediction_radius = (
            KeypointVisualizer._get_radius(prediction_canvas)
            if self.radius is None
            else self.radius
        )

        pred_viz = KeypointVisualizer.draw_predictions(
            pred_viz,
            keypoints,
            self.draw_indices,
            connectivity=self.connectivity,
            nonvisible_color=self.nonvisible_color,
            visible_color=self.visible_color,
            visibility_threshold=self.visibility_threshold,
            radius=prediction_radius,
            scale=self.scale,
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
                self.width,
                self.bbox_labels,
                self.colors,
                self.draw_labels,
                self.alpha,
                self.scale,
            )

        if target_keypoints is not None:
            target_radius = (
                KeypointVisualizer._get_radius(target_canvas)
                if self.radius is None
                else self.radius
            )
            target_viz = KeypointVisualizer.draw_targets(
                target_viz,
                target_keypoints,
                self.draw_indices,
                radius=target_radius,
                colors=self.visible_color,
                connectivity=self.connectivity,
            )

        return target_viz, pred_viz
