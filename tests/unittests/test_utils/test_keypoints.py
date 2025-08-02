import pytest
import torch

from luxonis_train.utils.keypoints import compute_pose_oks, get_sigmas


def test_get_sigmas():
    sigmas = [0.1, 0.2, 0.3]
    pytest.approx(get_sigmas(sigmas, 3).tolist(), sigmas)
    with pytest.raises(ValueError, match="length of the sigmas list"):
        get_sigmas(sigmas, 2)
    with pytest.raises(ValueError, match="test-caller-name"):
        get_sigmas(sigmas, 2, caller_name="test-caller-name")
    assert len(get_sigmas(None, 17)) == 17
    assert len(get_sigmas(None, 5)) == 5


def test_compute_pose_oks():
    with pytest.raises(ValueError, match="must be provided"):
        compute_pose_oks(
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            gt_bboxes=None,
            pose_area=None,
        )
