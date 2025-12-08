# Visualizers

> [!NOTE]
> **Important:** In all visualizations, the left side always displays the **Ground Truth (GT)** and the right side shows the **Prediction**.

> [!NOTE]
> 📐 **Looking for expected `predictions` and `targets` shapes for each task?**\
> Refer to the [**Prediction Shapes Overview**](../README.md) for the expected `prediction` formats, and see [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch) for details on the `target` formats.

Visualizers are used to render the output of a node. They are used in the `visualizers` field of the `Node` configuration.

## Table Of Contents

- [`BBoxVisualizer`](#bboxvisualizer)
- [`ClassificationVisualizer`](#classificationvisualizer)
- [`KeypointVisualizer`](#keypointvisualizer)
- [`SegmentationVisualizer`](#segmentationvisualizer)
- [`EmbeddingsVisualizer`](#embeddingsvisualizer)
- [`OCRVisualizer`](#ocrvisualizer)
- [`InstanceSegmentationVisualizer`](#instancesegmentationvisualizer)

## `BBoxVisualizer`

Visualizer for **bounding box detection task**.

**Parameters:**

| Key         | Type                                                                                  | Default value | Description                                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`    | `dict[int, str] \| list[str] \| None`                                                 | `None`        | Either a dictionary mapping class indices to names, or a list of names. If list is provided, the label mapping is done by index. By default, no labels are drawn        |
| `colors`    | `dict[int, tuple[int, int, int] \| str] \| list[tuple[int, int, int] \| str] \| None` | `None`        | Colors to use for the bounding boxes. Either a dictionary mapping class names to colors, or a list of colors. Color can be either a tuple of RGB values or a hex string |
| `fill`      | `bool`                                                                                | `False`       | Whether to fill the bounding boxes                                                                                                                                      |
| `width`     | `int`                                                                                 | `1`           | The width of the bounding box lines                                                                                                                                     |
| `font`      | `str \| None`                                                                         | `None`        | A filename containing a `TrueType` font                                                                                                                                 |
| `font_size` | `int \| None`                                                                         | `None`        | Font size used for the labels                                                                                                                                           |
| `scale`     | `float`                                                                               | `1.0`         | Scales the canvas and the annotations by a given factor                                                                                                                 |

**Example:**

![bounding_box_viz_example](../../../media/example_viz/bbox.png)

## `KeypointVisualizer`

Visualizer for **instance keypoint detection task**.

**Parameters:**

| Key                    | Type                                   | Default value | Description                                                                                                                     |
| ---------------------- | -------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `visibility_threshold` | `float`                                | `0.5`         | Threshold for visibility of keypoints. If the visibility of a keypoint is below this threshold, it is considered as not visible |
| `connectivity`         | `list[tuple[int, int]] \| None`        | `None`        | List of tuples of keypoint indices that define the connections in the skeleton                                                  |
| `visible_color`        | `str \| tuple[int, int, int]`          | `"red"`       | Color of visible keypoints                                                                                                      |
| `nonvisible_color`     | `str \| tuple[int, int, int ] \| None` | `None`        | Color of non-visible keypoints. If `None`, non-visible keypoints are not drawn                                                  |
| `radius`               | `int \| None`                          | `None`        | Radius of drawn keypoint dots. If `None`, dynamically determine this based on image dimensions                                  |
| `scale`                | `float`                                | `1.0`         | Scales the canvas and the annotations by a given factor                                                                         |
| `draw_indices`         | `bool`                                 | `False`       | Draw keypoint indices next to the keypoints                                                                                     |

**Example:**

![keypoints_viz_example](../../../media/example_viz/kpts.png)

## `SegmentationVisualizer`

Visualizer for **segmentation tasks**.

**Parameters:**

| Key     | Type                          | Default value | Description                                             |
| ------- | ----------------------------- | ------------- | ------------------------------------------------------- |
| `color` | `str \| tuple[int, int, int]` | `"#5050FF"`   | Color of the segmentation masks                         |
| `alpha` | `float`                       | `0.6`         | Alpha value of the segmentation masks                   |
| `scale` | `float`                       | `1.0`         | Scales the canvas and the annotations by a given factor |

**Example:**

![segmentation_viz_example](../../../media/example_viz/seg.png)

## `ClassificationVisualizer`

Visualizer for **classification tasks**.

**Parameters:**

| Key            | Type                   | Default value | Description                                                                      |
| -------------- | ---------------------- | ------------- | -------------------------------------------------------------------------------- |
| `include_plot` | `bool`                 | `True`        | Whether to include a plot of the class probabilities in the visualization        |
| `color`        | `tuple[int, int, int]` | `(255, 0, 0)` | Color of the text                                                                |
| `font_scale`   | `float`                | `1.0`         | Scale of the font                                                                |
| `thickness`    | `int`                  | `1`           | Line thickness of the font                                                       |
| `multi_label`  | `bool`                 | `False`       | Set to `True` for multi-label classification, otherwise `False` for single-label |
| `scale`        | `float`                | `1.0`         | Scales the canvas and the annotations by a given factor                          |

**Example:**

![class_viz_example](../../../media/example_viz/class.png)

## `EmbeddingsVisualizer`

Visualizer for **embedding tasks**.

**Parameters:**

| Key                 | Type    | Default value | Description                                                                                                                   |
| ------------------- | ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `z_score_threshold` | `float` | `3.0`         | Threshold for z-score filtering. Embeddings with z-score higher than this value are considered as outliers and are not drawn. |

**Example:**

![emb_viz_example](../../../media/example_viz/embeddings.png)

## `OCRVisualizer`

Visualizer for **OCR tasks**.

**Parameters:**

| Key          | Type                   | Default value | Description                                             |
| ------------ | ---------------------- | ------------- | ------------------------------------------------------- |
| `font_scale` | `float`                | `0.5`         | Font scale of the text. Defaults to `0.5`.              |
| `color`      | `tuple[int, int, int]` | `(0, 0, 0)`   | Color of the text. Defaults to `(0, 0, 0)`.             |
| `thickness`  | `int`                  | `1`           | Thickness of the text. Defaults to `1`.                 |
| `scale`      | `float`                | `1.0`         | Scales the canvas and the annotations by a given factor |

**Example:**

![ocr_viz_example](../../../media/example_viz/ocr.png)

## `InstanceSegmentationVisualizer`

Visualizer for **instance segmentation tasks**.

**Parameters:**

| Key         | Type                                                                                  | Default value | Description                                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`    | `dict[int, str] \| list[str] \| None`                                                 | `None`        | Either a dictionary mapping class indices to names, or a list of names. If list is provided, the label mapping is done by index. By default, no labels are drawn        |
| `colors`    | `dict[int, tuple[int, int, int] \| str] \| list[tuple[int, int, int] \| str] \| None` | `None`        | Colors to use for the bounding boxes. Either a dictionary mapping class names to colors, or a list of colors. Color can be either a tuple of RGB values or a hex string |
| `fill`      | `bool`                                                                                | `False`       | Whether to fill the bounding boxes                                                                                                                                      |
| `width`     | `int`                                                                                 | `1`           | The width of the bounding box lines                                                                                                                                     |
| `font`      | `str \| None`                                                                         | `None`        | A filename containing a `TrueType` font                                                                                                                                 |
| `font_size` | `int \| None`                                                                         | `None`        | Font size used for the labels                                                                                                                                           |
| `scale`     | `float`                                                                               | `1.0`         | Scales the canvas and the annotations by a given factor                                                                                                                 |

**Example:**

![instance_esg_viz_example](../../../media/example_viz/instance_seg.png)

## `FOMOVisualizer`

Visualizer for **FOMO object detection**.

**Parameters**:

| Key                    | Type    | Default value | Description                                                                                                             |
| ---------------------- | ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `visibility_threshold` | `float` | `0.5`         | Threshold for visibility of detected points. If the visibility of a point is below this threshold, it will not be shown |

**Example:**

![fomo_viz_example](../../../media/example_viz/fomo_parking_lot.png)
