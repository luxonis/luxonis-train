# Loaders

## Table Of Contents

- [`LuxonisLoaderTorch`](#luxonisloadertorch)
  - [Implementing a custom loader](#implementing-a-custom-loader)
- [`LuxonisLoaderPerlinNoise`](#luxonisloaderperlinnoise)

## `LuxonisLoaderTorch`

The default loader used with `LuxonisTrain`. It can either load data from an already created dataset in the `LuxonisDataFormat` or create a new dataset automatically from a set of supported formats.

**Parameters:**

| Key                   | Type                                                                                                      | Default value | Description                                                                                                                                                                                                                                                                                                |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_name`        | `str`                                                                                                     | `None`        | Name of the dataset to load. If not provided, the `dataset_dir` must be provided instead                                                                                                                                                                                                                   |
| `dataset_dir`         | `str`                                                                                                     | `None`        | Path to the directory containing the dataset. If not provided, the `dataset_name` must be provided instead. Can be a path to a local directory or a URL. The data can be in a zip archive. New `LuxonisDataset` will be created using data from this directory and saved under the provided `dataset_name` |
| `dataset_type`        | `Literal["coco", "voc", "darknet", "yolov6", "yolov4", "createml", "tfcsv", "clsdir", "segmask"] \| None` | `None`        | Type of the dataset. If not provided, the type will be inferred from the directory structure                                                                                                                                                                                                               |
| `team_id`             | `str \| None`                                                                                             | `None`        | Optional unique team identifier for the cloud                                                                                                                                                                                                                                                              |
| `bucket_storage`      | `Literal["local", "s3", "gcs"]`                                                                           | `"local"`     | Type of the bucket storage                                                                                                                                                                                                                                                                                 |
| `delete_existing`     | `bool`                                                                                                    | `False`       | Whether to delete the existing dataset with the same name. Only relevant if `dataset_dir` is provided. Use if you want to reparse the directory in case the data changed                                                                                                                                   |
| `update_mode`         | `Literal["all", "missing"]`                                                                               | `all`         | Select whether to download all remote dataset media files or only those missing locally                                                                                                                                                                                                                    |
| `min_bbox_visibility` | `float`                                                                                                   | `0.0`         | Minimum fraction of the original bounding box that must remain visible after augmentation.                                                                                                                       |
| `bbox_area_threshold` | `float`                                                                                                   | `0.0004`      | Minimum area threshold for bounding boxes to be considered valid (relative units in [0,1]). Boxes (and their related keypoints) with area below this will be filtered out.                                                                                                                                 |

**Data Shape Definitions:**

Below we list the output shapes of `LuxonisTorchLoader` returned by the `collate_fn` function. These are the shapes used during model training.

| Task Type               | Data Shape                                         | Description                                                                                                                                                   |
| ----------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bboxes`                | `torch.Tensor` of shape `(N, 6)`                   | N is the number of bounding boxes in the batch. The columns represent: batch index, class index, x-min, y-min, width, and height.                             |
| `keypoints`             | `torch.Tensor` of shape `(N, N_keypoints * 3 + 1)` | N is the number of keypoint sets in the batch. The first element is the batch index, followed by groups of three values (x, y, visibility) for each keypoint. |
| `segmentation`          | `torch.Tensor` of shape `(B, C, H, W)`             | B is the batch size; C is the number of classes; H and W are the height and width of the segmentation mask, respectively.                                     |
| `instance segmentation` | `torch.Tensor` of shape `(N, H, W)`                | N is the number of instance segmentation masks, ordered to correspond with the bounding boxes.                                                                |
| `ocr`                   | `torch.Tensor` of shape `(B, S)`                   | B is the batch size and S is the maximum sequence length, with zero-padding applied to standardize sequence lengths.                                          |
| `embeddings`            | `torch.Tensor` of shape `(B, K)`                   | B is the batch size and K is the dimensionality of the embedding vector.                                                                                      |
| `classification`        | `torch.Tensor` of shape `(B,)`                     | B is the batch size. Each element is the class index for the corresponding image in the batch.                                                                |

### Implementing a custom loader

To implement a loader, you need to create a class that inherits from [`BaseLoaderTorch`](./base_loader.py) and implement the following methods:

- `input_shapes(self) -> dict[str, torch.Size]`: Returns a dictionary with input shapes for each input image.
- `__len__(self) -> int`: Returns the number of samples in the dataset.
- `__getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]`: Returns a dictionary with input tensors for each input image.
- `get_classes(self) -> dict[str, list[str]]`: Returns a dictionary with class names for each task in the dataset.
- `collate_fn(self, batch: list[LuxonisLoaderTorchOutput]) -> tuple[dict[str, Tensor], Labels]`: A custom collation function that merges a list of samples into a single batch. It prepares inputs and labels in the format expected by the models during training.

For loaders yielding keypoint tasks, you also have to implement `get_n_keypoints(self) -> dict[str, int]` method.

For more information, consult the in-source [documentation](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/loaders/base_loader.py).

## LuxonisLoaderPerlinNoise

The loader used to train the [Unsupervised anomaly detection model](../config/predefined_models/README.md#anomalydetectionmodel).

| Key                   | Type            | Default value | Description                                                                                                                       |
| --------------------- | --------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `anomaly_source_path` | `str`           | `None`        | Path to the anomaly dataset from where random samples are drawn for noise. The DTD (Describable Textures Dataset) should be used. |
| `noise_prob`          | `float`         | `0.5`         | The probability with which to apply Perlin noise                                                                                  |
| `beta`                | `float \| None` | `None`        | The opacity of the anomaly mask. If None, a random value is chosen. It's advisable to set it to None.                             |

**Data Shape Definitions:**

| Task Type      | Data Shape                             | Description                                                                                                                                               |
| -------------- | -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `segmentation` | `torch.Tensor` of shape `(B, 2, H, W)` | B is the batch size; 2 because it includes the anomaly class and the background; H and W are the height and width of the segmentation mask, respectively. |
