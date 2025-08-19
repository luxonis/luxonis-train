import random
from typing import Any

import optuna
from loguru import logger


def _augs_to_indices(all_augs: list[str], aug_names: list[str]) -> list[int]:
    """Maps augmentation names to indices."""
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
        key_info = key.split("_")
        key_name = "_".join(key_info[:-1])
        key_type = key_info[-1]
        match key_type, value:
            case "subset", [list(whole_set), int(subset_size)]:
                if key_name.split(".")[-1] != "augmentations":
                    raise ValueError(
                        "Subset sampling currently only supported for augmentations"
                    )
                whole_set_indices = _augs_to_indices(all_augs, whole_set)
                subset = random.sample(whole_set_indices, subset_size)
                for aug_id in whole_set_indices:
                    new_params[f"{key_name}.{aug_id}.active"] = (
                        aug_id in subset
                    )
                continue
            case "categorical", list(lst):
                new_value = trial.suggest_categorical(key_name, lst)
            case "float", [float(low), float(high), *tail]:
                step = tail[0] if tail else None
                if step is not None and not isinstance(step, float):
                    raise ValueError(
                        f"Step for float type must be float, but got {step}"
                    )
                new_value = trial.suggest_float(key_name, low, high, step=step)
            case "int", [int(low), int(high), *tail]:
                step = tail[0] if tail else 1
                if not isinstance(step, int):
                    raise TypeError(
                        f"Step for int type must be int, but got {step}"
                    )
                new_value = trial.suggest_int(key_name, low, high, step=step)
            case "loguniform", [float(low), float(high)]:
                new_value = trial.suggest_loguniform(key_name, low, high)
            case "uniform", [float(low), float(high)]:
                new_value = trial.suggest_uniform(key_name, low, high)
            case _, _:
                raise KeyError(
                    f"Combination of {key_type} and {value} not supported"
                )

        new_params[key_name] = new_value

    if len(new_params) == 0:
        raise ValueError(
            "No paramteres to tune. Specify them under `tuner.params`."
        )
    return new_params


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
