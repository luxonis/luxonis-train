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


def test_subset_outside_augmentations_raises():
    with pytest.raises(ValueError, match="only supported for augmentations"):
        get_trial_params(
            [], {"trainer.optimizer.params.lr_subset": [["a"], 1]}, _trial()
        )


def test_malformed_augmentation_subset_raises():
    with pytest.raises(KeyError, match="Combination of subset"):
        get_trial_params(
            [],
            {"trainer.preprocessing.augmentations_subset": [0.1, 0.5]},
            _trial(),
        )


def test_suggests_loguniform_and_uniform():
    params = get_trial_params(
        [],
        {
            "trainer.optimizer.params.lr_loguniform": [0.0001, 0.1],
            "trainer.optimizer.params.momentum_uniform": [0.1, 0.5],
        },
        _trial(),
    )
    assert 0.0001 <= params["trainer.optimizer.params.lr"] <= 0.1
    assert 0.1 <= params["trainer.optimizer.params.momentum"] <= 0.5


def test_float_step_must_be_float():
    with pytest.raises(ValueError, match="Step for float type must be float"):
        get_trial_params(
            [], {"trainer.optimizer.params.lr_float": [0.1, 0.5, 1]}, _trial()
        )


def test_int_step_must_be_int():
    with pytest.raises(TypeError, match="Step for int type must be int"):
        get_trial_params([], {"trainer.batch_size_int": [2, 8, 2.5]}, _trial())


def test_unknown_key_type_raises():
    with pytest.raises(KeyError, match="Combination of bogus"):
        get_trial_params([], {"trainer.foo_bogus": [1, 2]}, _trial())


def test_numeric_value_must_be_a_pair():
    with pytest.raises(KeyError, match="Combination of int"):
        get_trial_params([], {"trainer.batch_size_int": 5}, _trial())


def test_numeric_bounds_must_match_the_type():
    with pytest.raises(KeyError, match="Combination of int"):
        get_trial_params([], {"trainer.batch_size_int": [2.5, 8.5]}, _trial())


def _trial() -> optuna.trial.Trial:
    return optuna.create_study().ask()
