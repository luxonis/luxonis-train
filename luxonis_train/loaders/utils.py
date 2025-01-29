import torch
from luxonis_ml.data.utils import get_task_type, task_is_metadata
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
    @rtype: tuple[dict[str, Tensor], dict[str, Tensor]]
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
            for i, ann in enumerate(annos):
                new_ann = torch.zeros((ann.shape[0], ann.shape[1] + 1))
                # add batch index to separate boxes from different images
                new_ann[:, 0] = i
                new_ann[:, 1:] = ann
                label_box.append(new_ann)
            out_labels[task] = torch.cat(label_box, 0)
        elif task_type == "instance_segmentation":
            out_labels[task] = torch.cat(annos, 0)
        elif task_is_metadata(task):
            if task_type == "metadata/text":
                max_len = max(len(anno) for anno in annos)
                padded_annos = torch.zeros(
                    len(annos), max_len, dtype=torch.int32
                )
                for i, anno in enumerate(annos):
                    padded_annos[i, : len(anno)] = anno
                out_labels[task] = padded_annos
            else:
                out_labels[task] = torch.cat(annos, 0)
        else:
            out_labels[task] = torch.stack(annos, 0)

    return out_inputs, out_labels
