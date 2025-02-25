# Visualizers

> \[!NOTE\]
> **Important:** In all visualizations, the left side always displays the **Ground Truth (GT)** and the right side shows the **Prediction**.

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

Visualizer for bounding box detection task.

**Parameters:**

| Key         | Type                                                                                  | Default value | Description                                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`    | `dict[int, str] \| list[str] \| None`                                                 | `None`        | Either a dictionary mapping class indices to names, or a list of names. If list is provided, the label mapping is done by index. By default, no labels are drawn        |
| `colors`    | `dict[int, tuple[int, int, int] \| str] \| list[tuple[int, int, int] \| str] \| None` | `None`        | Colors to use for the bounding boxes. Either a dictionary mapping class names to colors, or a list of colors. Color can be either a tuple of RGB values or a hex string |
| `fill`      | `bool`                                                                                | `False`       | Whether to fill the bounding boxes                                                                                                                                      |
| `width`     | `int`                                                                                 | `1`           | The width of the bounding box lines                                                                                                                                     |
| `font`      | `str \| None`                                                                         | `None`        | A filename containing a `TrueType` font                                                                                                                                 |
| `font_size` | `int \| None`                                                                         | `None`        | Font size used for the labels                                                                                                                                           |

The `BBoxVisualizer` uses predictions as a list of \[N, 6\] tensors, where each tensor in the list corresponds to one image. Each row in the tensor represents a detected object and contains the following columns:

- `x_min`: The x-coordinate of the top-left corner of the bounding box.
- `y_min`: The y-coordinate of the top-left corner of the bounding box.
- `x_max`: The x-coordinate of the bottom-right corner of the bounding box.
- `y_max`: The y-coordinate of the bottom-right corner of the bounding box.
- `pred_score`: The confidence score of the prediction.
- `pred_cls`: The predicted class index of the detected object.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![bounding_box_viz_example](../../../media/example_viz/bbox.png)

## `KeypointVisualizer`

Visualizer for instance keypoint detection task.

**Parameters:**

| Key                    | Type                                   | Default value | Description                                                                                                                     |
| ---------------------- | -------------------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `visibility_threshold` | `float`                                | `0.5`         | Threshold for visibility of keypoints. If the visibility of a keypoint is below this threshold, it is considered as not visible |
| `connectivity`         | `list[tuple[int, int]] \| None`        | `None`        | List of tuples of keypoint indices that define the connections in the skeleton                                                  |
| `visible_color`        | `str \| tuple[int, int, int]`          | `"red"`       | Color of visible keypoints                                                                                                      |
| `nonvisible_color`     | `str \| tuple[int, int, int ] \| None` | `None`        | Color of non-visible keypoints. If `None`, non-visible keypoints are not drawn                                                  |

The `KeypointVisualizer` uses predictions in the form of a list of \[N, N_keypoints, 3\] tensors, where each tensor in the list corresponds to one image. Each tensor entry includes:

- `x`: The x-coordinate of the keypoint.
- `y`: The y-coordinate of the keypoint.
- `visibility`: The visibility score of the keypoint detection.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![keypoints_viz_example](../../../media/example_viz/kpts.png)

## `SegmentationVisualizer`

Visualizer for segmentation tasks.

**Parameters:**

| Key     | Type                          | Default value | Description                           |
| ------- | ----------------------------- | ------------- | ------------------------------------- |
| `color` | `str \| tuple[int, int, int]` | `"#5050FF"`   | Color of the segmentation masks       |
| `alpha` | `float`                       | `0.6`         | Alpha value of the segmentation masks |

The `SegmentationVisualizer` uses predictions in the form of a tensor with shape `[B, C, H, W]`, where:

- `B` is the batch size, representing the number of images in the batch.
- `C` is the number of channels, each representing a class for segmentation.
- `H` is the height of the segmentation masks.
- `W` is the width of the segmentation masks.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![segmentation_viz_example](../../../media/example_viz/seg.png)

## `ClassificationVisualizer`

Visualizer for classification tasks.

**Parameters:**

| Key            | Type                   | Default value | Description                                                                      |
| -------------- | ---------------------- | ------------- | -------------------------------------------------------------------------------- |
| `include_plot` | `bool`                 | `True`        | Whether to include a plot of the class probabilities in the visualization        |
| `color`        | `tuple[int, int, int]` | `(255, 0, 0)` | Color of the text                                                                |
| `font_scale`   | `float`                | `1.0`         | Scale of the font                                                                |
| `thickness`    | `int`                  | `1`           | Line thickness of the font                                                       |
| `multi_label`  | `bool`                 | `False`       | Set to `True` for multi-label classification, otherwise `False` for single-label |

The `ClassificationVisualizer` uses predictions in the form of a tensor with shape `[B]`, where:

- `B` represents the batch size, and each element in the tensor is the predicted class index for each input example in the batch.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![class_viz_example](../../../media/example_viz/class.png)

## `EmbeddingsVisualizer`

Visualizer for embedding tasks.

**Parameters:**

| Key                 | Type    | Default value | Description                                                                                                                   |
| ------------------- | ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `z_score_threshold` | `float` | `3.0`         | Threshold for z-score filtering. Embeddings with z-score higher than this value are considered as outliers and are not drawn. |

The `EmbeddingsVisualizer` uses predictions in the form of a tensor with shape `[B, F]`, where:

- `B` is the batch size, representing the number of data points.
- `F` is the feature dimension of each embedding vector.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![emb_viz_example](../../../media/example_viz/embeddings.png)

## `OCRVisualizer`

Visualizer for OCR tasks.

**Parameters:**

| Key          | Type                   | Default value | Description                                 |
| ------------ | ---------------------- | ------------- | ------------------------------------------- |
| `font_scale` | `float`                | `0.5`         | Font scale of the text. Defaults to `0.5`.  |
| `color`      | `tuple[int, int, int]` | `(0, 0, 0)`   | Color of the text. Defaults to `(0, 0, 0)`. |
| `thickness`  | `int`                  | `1`           | Thickness of the text. Defaults to `1`.     |

The `OCRVisualizer` uses predictions in the form of a list or tensor with shape `[B, seq_length]`, where:

- `B` is the batch size, each row corresponds to the encoded predictions for a different sample.
- `seq_length` represents the length of the prediction sequence for each sample.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![ocr_viz_example](../../../media/example_viz/ocr.png)

## `InstanceSegmentationVisualizer`

Visualizer for instance segmentation tasks.

**Parameters:**

| Key         | Type                                                                                  | Default value | Description                                                                                                                                                             |
| ----------- | ------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `labels`    | `dict[int, str] \| list[str] \| None`                                                 | `None`        | Either a dictionary mapping class indices to names, or a list of names. If list is provided, the label mapping is done by index. By default, no labels are drawn        |
| `colors`    | `dict[int, tuple[int, int, int] \| str] \| list[tuple[int, int, int] \| str] \| None` | `None`        | Colors to use for the bounding boxes. Either a dictionary mapping class names to colors, or a list of colors. Color can be either a tuple of RGB values or a hex string |
| `fill`      | `bool`                                                                                | `False`       | Whether to fill the bounding boxes                                                                                                                                      |
| `width`     | `int`                                                                                 | `1`           | The width of the bounding box lines                                                                                                                                     |
| `font`      | `str \| None`                                                                         | `None`        | A filename containing a `TrueType` font                                                                                                                                 |
| `font_size` | `int \| None`                                                                         | `None`        | Font size used for the labels                                                                                                                                           |

The `InstanceSegmentationVisualizer` uses predictions in the form of a list of tensors, each with the shape `[N, H, W]`. Each tensor in the list corresponds to one image, where:

- `N` is the number of instance segmentation masks in the image.
- `H` is the height of each segmentation mask.
- `W` is the width of each segmentation mask.

Targets are in the format specified by the [`LuxonisLoaderTorch`](../../loaders/README.md#luxonisloadertorch).

**Example:**

![instance_esg_viz_example](../../../media/example_viz/instance_seg.png)
