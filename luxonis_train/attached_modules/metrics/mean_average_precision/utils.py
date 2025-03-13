from torch import Tensor
from torchvision.ops import box_convert


def compute_update_lists(
    predictions: list[Tensor], targets: Tensor, height: int, width: int
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    output_list: list[dict[str, Tensor]] = []
    label_list: list[dict[str, Tensor]] = []
    for i in range(len(predictions)):
        pred = {
            "boxes": predictions[i][:, :4],
            "scores": predictions[i][:, 4],
            "labels": predictions[i][:, 5].int(),
        }
        output_list.append(pred)

        curr_label = targets[targets[:, 0] == i]
        curr_bboxs = box_convert(curr_label[:, 2:], "xywh", "xyxy")
        curr_bboxs[:, 0::2] *= width
        curr_bboxs[:, 1::2] *= height

        gt = {
            "boxes": curr_bboxs,
            "labels": curr_label[:, 1].int(),
        }
        label_list.append(gt)

    return output_list, label_list
