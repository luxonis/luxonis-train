# Prediction Shapes Overview

> [!NOTE]
> For some custom losses, this may not apply, as they might require additional outputs from the head.\
> However, for metrics and visualizers, this table holds as is.

This table summarizes the expected shapes and formats for `predictions` for each default task in **luxonis-train**. Use it as a quick reference for implementing custom metrics, visualizers, or debugging shape mismatches.

For the expected shapes and formats of `targets`, see [`LuxonisLoaderTorch`](../loaders/README.md#luxonisloadertorch).

## Prediction Shape Reference

| Task (`Tasks.X`)              | Prediction Type               | Shape / Format        | Notes                                                                                              |
| ----------------------------- | ----------------------------- | --------------------- | -------------------------------------------------------------------------------------------------- |
| `Tasks.CLASSIFICATION`        | `torch.Tensor`                | `[B]`                 | 1D tensor of length `B`: one class index prediction per sample.                                    |
| `Tasks.BOUNDINGBOX`           | `List[torch.Tensor]`          | `[N_instances, 6]`    | List of tensors (one per image). Each row: `[x_min, y_min, x_max, y_max, score, class_idx]`.       |
| `Tasks.INSTANCE_KEYPOINTS`    | `List[torch.Tensor]`          | `[N_instances, K, 3]` | List of tensors (one per image). Last dim: `(x, y, visibility)` for each of the `K` keypoints.     |
| `Tasks.SEGMENTATION`          | `torch.Tensor`                | `[B, C, H, W]`        | 4D tensor: `B` images, `C` class channels, each of size `H×W`.                                     |
| `Tasks.EMBEDDINGS`            | `torch.Tensor`                | `[B, F]`              | 2D tensor of embeddings: `B` samples × `F`-dimensional feature vectors.                            |
| `Tasks.OCR`                   | `torch.Tensor` or `List[...]` | `[B, S]`              | Either a single tensor or a list per image. Each row is a sequence of length `S` (encoded tokens). |
| `Tasks.INSTANCE_SEGMENTATION` | `List[torch.Tensor]`          | `[N_instances, H, W]` | List of tensors (one per image). Each tensor contains `N_instances` binary masks of size `H×W`.    |
