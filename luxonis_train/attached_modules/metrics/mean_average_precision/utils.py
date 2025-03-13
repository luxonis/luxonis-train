from luxonis_ml.typing import all_not_none, any_not_none
from torch import Tensor
from torchvision.ops import box_convert


def compute_update_lists(
    boundinbox: list[Tensor],
    targets: Tensor,
    height: int,
    width: int,
    *,
    keypoints: list[Tensor] | None = None,
    n_keypoints: int | None = None,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    if any_not_none([keypoints, n_keypoints]) and not all_not_none(
        [keypoints, n_keypoints]
    ):
        raise ValueError(
            "Either both `keypoints` and `n_keypoints` "
            "must be provided, or neither."
        )
    output_list: list[dict[str, Tensor]] = []
    label_list: list[dict[str, Tensor]] = []
    for i in range(len(boundinbox)):
        pred = {
            "boxes": boundinbox[i][:, :4],
            "scores": boundinbox[i][:, 4],
            "labels": boundinbox[i][:, 5].int(),
        }
        if keypoints is not None and n_keypoints is not None:
            pred["keypoints"] = keypoints[i].reshape(-1, n_keypoints * 3)

        output_list.append(pred)

        target = targets[targets[:, 0] == i]
        bboxs = box_convert(target[:, 2:6], "xywh", "xyxy")
        bboxs[:, 0::2] *= width
        bboxs[:, 1::2] *= height

        gt = {
            "boxes": bboxs,
            "labels": target[:, 1].int(),
        }
        label_list.append(gt)

    return output_list, label_list
