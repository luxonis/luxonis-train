import optuna
import pytest

from luxonis_train.core.utils.tune_utils import get_trial_params


def test_missing_type_suffix_raises_key_error():
    with pytest.raises(
        KeyError, match=r"Combination of lr and \[0.1, 0.5\] not supported"
    ):
        get_trial_params([], {"lr": [0.1, 0.5]}, _trial())


def test_suggests_float_int_and_categorical():
    params = get_trial_params(
        [],
        {
            "trainer.optimizer.params.lr_float": [0.1, 0.5],
            "trainer.batch_size_int": [2, 8, 2],
            "trainer.precision_categorical": ["16", "32"],
        },
        _trial(),
    )
    assert 0.1 <= params["trainer.optimizer.params.lr"] <= 0.5
    assert params["trainer.batch_size"] in {2, 4, 6, 8}
    assert params["trainer.precision"] in {"16", "32"}


def test_empty_params_raise():
    with pytest.raises(ValueError, match="No parameters to tune"):
        get_trial_params([], {}, _trial())


def _trial() -> optuna.trial.Trial:
    return optuna.create_study().ask()
