import random
from typing import Any, TypeGuard

import optuna
from loguru import logger


def _augs_to_indices(all_augs: list[str], aug_names: list[str]) -> list[int]:
    """Map augmentation names to indices."""
    aug_indices = []
    for aug_name in aug_names:
        if aug_name == "Normalize":
            logger.warning(
                f"'{aug_name}' should be tuned directly by adding '...normalize.active_categorical' to the tuner params, skipping."
            )
            continue
        try:
            index = all_augs.index(aug_name)
            aug_indices.append(index)
        except ValueError:
            logger.warning(
                f"Augmentation '{aug_name}' not found under trainer augemntations, skipping."
            )
            continue
    return aug_indices


def get_trial_params(
    all_augs: list[str], params: dict[str, Any], trial: optuna.trial.Trial
) -> dict[str, Any]:
    """Get trial parameters based on specified config."""
    new_params = {}
    for key, value in params.items():
        key_name, _, key_type = key.rpartition("_")
        if key_type == "subset":
            new_params.update(
                _sample_augmentation_subset(all_augs, key_name, value)
            )
            continue
        new_params[key_name] = _suggest_trial_value(
            trial, key_name, key_type, value
        )

    if len(new_params) == 0:
        raise ValueError(
            "No parameters to tune. Specify them under `tuner.params`."
        )
    return new_params


def _sample_augmentation_subset(
    all_augs: list[str], key_name: str, value: object
) -> dict[str, bool]:
    if key_name.rsplit(".", 1)[-1] != "augmentations":
        raise ValueError(
            "Subset sampling currently only supported for augmentations"
        )
    if not (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], list)
        and isinstance(value[1], int)
    ):
        raise _unsupported_combination("subset", value)
    indices = _augs_to_indices(all_augs, value[0])
    selected = set(random.sample(indices, value[1]))
    return {
        f"{key_name}.{index}.active": index in selected for index in indices
    }


def _suggest_trial_value(
    trial: optuna.trial.Trial, key_name: str, key_type: str, value: object
) -> float | int | str | bool | None:
    if key_type == "categorical" and isinstance(value, list):
        return trial.suggest_categorical(key_name, value)
    if key_type in {"float", "int"}:
        return _suggest_numeric_value(trial, key_name, key_type, value)
    if key_type == "loguniform" and _is_pair_of_floats(value):
        return trial.suggest_loguniform(key_name, *value)
    if key_type == "uniform" and _is_pair_of_floats(value):
        return trial.suggest_uniform(key_name, *value)
    raise _unsupported_combination(key_type, value)


def _suggest_numeric_value(
    trial: optuna.trial.Trial, key_name: str, key_type: str, value: object
) -> float | int:
    if not isinstance(value, list) or len(value) < 2:
        raise _unsupported_combination(key_type, value)
    low, high, *tail = value
    if (
        key_type == "float"
        and isinstance(low, float)
        and isinstance(high, float)
    ):
        step = tail[0] if tail else None
        if step is not None and not isinstance(step, float):
            raise ValueError(
                f"Step for float type must be float, but got {step}"
            )
        return trial.suggest_float(key_name, low, high, step=step)
    if key_type == "int" and isinstance(low, int) and isinstance(high, int):
        step = tail[0] if tail else 1
        if not isinstance(step, int):
            raise TypeError(f"Step for int type must be int, but got {step}")
        return trial.suggest_int(key_name, low, high, step=step)
    raise _unsupported_combination(key_type, value)


def _is_pair_of_floats(value: object) -> TypeGuard[list[float]]:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, float) for item in value)
    )


def _unsupported_combination(key_type: str, value: object) -> KeyError:
    return KeyError(f"Combination of {key_type} and {value} not supported")


def rename_params_for_logging(
    params: dict, tuner_params: dict | None = None
) -> dict:
    """Rename parameters used for logging."""
    aug_subset = []
    if tuner_params:
        aug_subset, _ = tuner_params.get(
            "trainer.preprocessing.augmentations_subset", ([], [])
        )

    renamed = {}
    for k, v in params.items():
        if k.startswith("trainer.preprocessing.augmentations.") and aug_subset:
            parts = k.split(".")
            try:
                idx = int(parts[3])  # augmentations.<index>.<field>
                aug_name = aug_subset[idx]
                new_key = (
                    f"trainer.preprocessing.augmentations.{aug_name}.active"
                )
                renamed[new_key] = v
            except (IndexError, ValueError):
                renamed[k] = v
        else:
            renamed[k] = v
    return renamed
