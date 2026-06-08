import torch

from luxonis_train.utils.segmentation import seg_output_to_bool


def test_seg_output_to_bool_binary():
    output = torch.tensor([[[0.0, 2.0], [-2.0, 0.1]]])

    mask = seg_output_to_bool(output, binary_threshold=0.5)

    assert mask.tolist() == [[[True, True], [False, True]]]


def test_seg_output_to_bool_multiclass():
    output = torch.tensor(
        [
            [[0.9, 0.1], [0.1, 0.1]],
            [[0.1, 0.8], [0.2, 0.1]],
            [[0.0, 0.1], [0.7, 0.4]],
        ]
    )

    masks = seg_output_to_bool(output)

    assert masks.tolist() == [
        [[True, False], [False, False]],
        [[False, True], [False, False]],
        [[False, False], [True, True]],
    ]
