# Loaders

## Table Of Contents

- [`LuxonisLoaderTorch`](#luxonisloadertorch)
  - [Implementing a custom loader](#implementing-a-custom-loader)

## `LuxonisLoaderTorch`

The default loader used with `LuxonisTrain`. It can either load data from an already created dataset in the `LuxonisDataFormat` or create a new dataset automatically from a set of supported formats.

**Parameters:**

| Key               | Type                                                                                                      | Default value | Description                                                                                                                                                                                                                                                                                                |
| ----------------- | --------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_name`    | `str`                                                                                                     | `None`        | Name of the dataset to load. If not provided, the `dataset_dir` must be provided instead                                                                                                                                                                                                                   |
| `dataset_dir`     | `str`                                                                                                     | `None`        | Path to the directory containing the dataset. If not provided, the `dataset_name` must be provided instead. Can be a path to a local directory or a URL. The data can be in a zip archive. New `LuxonisDataset` will be created using data from this directory and saved under the provided `dataset_name` |
| `dataset_type`    | `Literal["coco", "voc", "darknet", "yolov6", "yolov4", "createml", "tfcsv", "clsdir", "segmask"] \| None` | `None`        | Type of the dataset. If not provided, the type will be inferred from the directory structure                                                                                                                                                                                                               |
| `team_id`         | `str \| None`                                                                                             | `None`        | Optional unique team identifier for the cloud                                                                                                                                                                                                                                                              |
| `bucket_storage`  | `Literal["local", "s3", "gcs"]`                                                                           | `"local"`     | Type of the bucket storage                                                                                                                                                                                                                                                                                 |
| `delete_existing` | `bool`                                                                                                    | `False`       | Whether to delete the existing dataset with the same name. Only relevant if `dataset_dir` is provided. Use if you want to reparse the directory in case the data changed                                                                                                                                   |

### Implementing a custom loader

To implement a loader, you need to create a class that inherits from `BaseLoaderTorch` and implement the following methods:

- `input_shapes(self) -> dict[str, torch.Size]`: Returns a dictionary with input shapes for each input image.
- `__len__(self) -> int`: Returns the number of samples in the dataset.
- `__getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, luxonis_train.enums.TaskType]]`: Returns a dictionary with input tensors for each input image.
- `get_classes(self) -> dict[str, list[str]]`: Returns a dictionary with class names for each task in the dataset.

For loaders yielding keypoint tasks, you also have to implement `get_n_keypoints(self) -> dict[str, int]` method.

For more information, consult the in-source [documentation](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/loaders/base_loader.py).
