import torch
from luxonis_ml.data.utils import get_task_type
from torch import Tensor

from luxonis_train.utils.types import Labels

LuxonisLoaderTorchOutput = tuple[dict[str, Tensor], Labels]
"""LuxonisLoaderTorchOutput is a tuple of source tensors and
corresponding labels."""


def collate_fn(
    batch: list[LuxonisLoaderTorchOutput],
) -> tuple[dict[str, Tensor], Labels]:
    """Default collate function used for training.

    @type batch: list[LuxonisLoaderTorchOutput]
    @param batch: List of loader outputs (dict of Tensors) and labels
        (dict of Tensors) in the LuxonisLoaderTorchOutput format.
    @rtype: tuple[dict[str, Tensor], dict[TaskType, Tensor]]
    @return: Tuple of inputs and annotations in the format expected by
        the model.
    """
    inputs: tuple[dict[str, Tensor], ...]
    labels: tuple[Labels, ...]
    inputs, labels = zip(*batch)

    out_inputs = {
        k: torch.stack([i[k] for i in inputs], 0) for k in inputs[0].keys()
    }

    out_labels: Labels = {}

    for task in labels[0].keys():
        task_type = get_task_type(task)
        annos = [label[task] for label in labels]

        if task_type in {"keypoints", "boundingbox"}:
            label_box: list[Tensor] = []
            for i, box in enumerate(annos):
                l_box = torch.zeros((box.shape[0], box.shape[1] + 1))
                l_box[:, 0] = i  # add target image index for build_targets()
                l_box[:, 1:] = box
                label_box.append(l_box)
            out_labels[task] = torch.cat(label_box, 0)

        elif task_type == "instance_segmentation":
            masks = [label[task] for label in labels]
            out_labels[task] = torch.cat(masks, 0)
        else:
            out_labels[task] = torch.stack(annos, 0)

    return out_inputs, out_labels
