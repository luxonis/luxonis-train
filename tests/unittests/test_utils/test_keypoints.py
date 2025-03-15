import pytest

from luxonis_train.utils.keypoints import get_sigmas


def test_get_sigmas():
    sigmas = [0.1, 0.2, 0.3]
    pytest.approx(get_sigmas(sigmas, 3).tolist(), sigmas)
    with pytest.raises(ValueError, match="length of the sigmas list"):
        get_sigmas(sigmas, 2)
    assert len(get_sigmas(None, 17)) == 17
    assert len(get_sigmas(None, 5)) == 5
