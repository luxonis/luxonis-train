# Loaders

## LuxonisLoaderTorch

The default loader used with `LuxonisTrain`. It can either load data from an already created dataset in the `LuxonisDataFormat` or create a new dataset automatically from a set of supported formats.

### Implementing a custom loader

To implement a loader, you need to create a class that inherits from `BaseLoaderTorch` and implement the following methods:

- `input_shapes(self) -> dict[str, torch.Size]`: Returns a dictionary with input shapes for each input image.
- `__len__(self) -> int`: Returns the number of samples in the dataset.
- `__getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, luxonis_train.enums.TaskType]]`: Returns a dictionary with input tensors for each input image.
- `get_classes(self) -> dict[str, list[str]]`: Returns a dictionary with class names for each task in the dataset.

For loaders yielding keypoint tasks, you also have to implement `get_n_keypoints(self) -> dict[str, int]` method.

For more information, consult the in-source [documentation](https://github.com/luxonis/luxonis-train/blob/main/luxonis_train/loaders/base_loader.py).
