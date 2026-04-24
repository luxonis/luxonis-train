from luxonis_train.config.config import (
    NormalizeAugmentationConfig,
    PreprocessingConfig,
)
from luxonis_train.core.utils.export_utils import get_preprocessing


def test_get_preprocessing_skips_inactive_normalization():
    cfg = PreprocessingConfig(
        color_space="BGR",
        normalize=NormalizeAugmentationConfig(
            active=False,
            params={
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        ),
    )

    mean, scale, color_space = get_preprocessing(
        cfg, "Exporting to NN Archive"
    )

    assert mean is None
    assert scale is None
    assert color_space == "BGR"


def test_get_preprocessing_returns_scaled_active_normalization():
    cfg = PreprocessingConfig(
        normalize=NormalizeAugmentationConfig(
            active=True,
            params={
                "mean": [0.5, 0.25, 0.125],
                "std": [0.1, 0.2, 0.4],
            },
        ),
    )

    mean, scale, color_space = get_preprocessing(cfg)

    assert mean == [127.5, 63.75, 31.875]
    assert scale == [25.5, 51.0, 102.0]
    assert color_space == "RGB"
