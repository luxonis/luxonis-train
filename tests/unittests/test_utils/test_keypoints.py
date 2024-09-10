import pytest
import torch

from luxonis_train.utils.keypoints import get_sigmas, process_keypoints_predictions


def test_get_sigmas():
    sigmas = [0.1, 0.2, 0.3]
    pytest.approx(get_sigmas(sigmas, 3).tolist(), sigmas)
    with pytest.raises(ValueError):
        get_sigmas(sigmas, 2)
    assert len(get_sigmas(None, 17)) == 17
    assert len(get_sigmas(None, 5)) == 5


def test_process_keypoints_predictions():
    keypoints = torch.tensor([[0.1, 0.2, 1.0, 0.4, 0.5, 0.0]])
    x, y, visibility = process_keypoints_predictions(keypoints)
    pytest.approx(x[0].tolist(), [0.1, 0.4])
    pytest.approx(y[0].tolist(), [0.2, 0.5])
    pytest.approx(visibility[0].tolist(), [1.0, 0.0])
