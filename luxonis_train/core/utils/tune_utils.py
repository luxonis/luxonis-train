"""Translation between the ``tuner.params`` of a config and the
suggestions an Optuna trial makes.
"""

import random
from typing import Any, TypeGuard

import optuna
from loguru import logger


def _augs_to_indices(all_augs: list[str], aug_names: list[str]) -> list[int]:
    """Map augmentation names to their indices in ``all_augs``.

    The function skips ``Normalize`` and a name that ``all_augs`` does
    not hold, and logs a warning for each.

    Args:
        all_augs (list[str]): The names of the augmentations in
            ``trainer.preprocessing.augmentations``, in config order.
        aug_names (list[str]): The names to map.

    Returns:
        list[int]: The index of each name that the function keeps, in
        the order of ``aug_names``.

    """
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
    """Sample the config overrides of one trial.

    Each key of ``params`` is a dotted config key with a type suffix,
    such as ``trainer.optimizer.params.lr_float``. The suffix after the
    last ``_`` selects the sampling:

    - ``categorical``: ``trial.suggest_categorical`` from a list.
    - ``int``: ``trial.suggest_int`` from ``[low, high]`` or
      ``[low, high, step]``, as integers. The default step is ``1``.
    - ``float``: ``trial.suggest_float`` from ``[low, high]`` or
      ``[low, high, step]``, as floats.
    - ``uniform``: ``trial.suggest_float`` from ``[low, high]``, as
      floats.
    - ``loguniform``: ``trial.suggest_float`` from ``[low, high]``, as
      floats, with ``log=True``.
    - ``subset``: only for a key whose last dotted part is
      ``augmentations``. The value is a list of augmentation names and
      a count. The function picks that many of the names at random with
      the ``random`` module, not with ``trial``. It skips ``Normalize``
      and unknown names with a warning.

    The name of an Optuna parameter is the key without the suffix.

    Args:
        all_augs (list[str]): The names of the augmentations in
            ``trainer.preprocessing.augmentations``, in config order.
            A ``subset`` key uses them to find the indices.
        params (``dict[str, Any]``): The ``tuner.params`` section of the
            config.
        trial (``optuna.trial.Trial``): The trial that samples the
            values.

    Returns:
        ``dict[str, Any]``: The sampled value of each key without the
        suffix. A ``subset`` key gives one boolean entry
        ``<key>.<index>.active`` for each kept augmentation name.
        ``True`` marks a picked augmentation.

    Raises:
        ValueError: When the result has no entry, for example for an
            empty ``params``. Also when the last dotted part of a
            ``subset`` key is not ``augmentations``, when the count of a
            ``subset`` key is larger than the number of kept names, or
            when the step of a ``float`` key is not a float.
        TypeError: When the step of an ``int`` key is not an integer.
        KeyError: When a suffix is unknown, or when a value does not fit
            its suffix.

    Example:
        >>> import optuna
        >>> trial = optuna.trial.FixedTrial(
        ...     {"trainer.batch_size": 8, "trainer.optimizer.name": "SGD"}
        ... )
        >>> params = {
        ...     "trainer.batch_size_int": [4, 16, 4],
        ...     "trainer.optimizer.name_categorical": ["Adam", "SGD"],
        ... }
        >>> get_trial_params([], params, trial)
        {'trainer.batch_size': 8, 'trainer.optimizer.name': 'SGD'}

    """
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


def rename_params_for_logging(
    params: dict, tuner_params: dict | None = None
) -> dict:
    """Replace the augmentation indices in the keys with names, for logs.

    The function reads the list of names of
    ``trainer.preprocessing.augmentations_subset`` in ``tuner_params``.
    A key ``trainer.preprocessing.augmentations.<index>.<field>``
    becomes ``trainer.preprocessing.augmentations.<name>.active``, where
    ``<name>`` is the entry at ``<index>`` in that list. A key keeps its
    name when ``<index>`` is not an integer or is out of range for the
    list. The other keys keep their names too.

    Args:
        params (dict): The sampled parameters of a trial, as
            `get_trial_params` returns them.
        tuner_params (dict | None): The ``tuner.params`` section of the
            config. Without a ``subset`` entry for the augmentations,
            the function changes no key.

    Returns:
        dict: A new dictionary with the same values as ``params``.

    Example:
        >>> params = {
        ...     "trainer.preprocessing.augmentations.1.active": False,
        ...     "trainer.batch_size": 8,
        ... }
        >>> tuner_params = {
        ...     "trainer.preprocessing.augmentations_subset": [
        ...         ["Defocus", "Sharpen"],
        ...         1,
        ...     ]
        ... }
        >>> rename_params_for_logging(params, tuner_params)
        {'trainer.preprocessing.augmentations.Sharpen.active': False,
         'trainer.batch_size': 8}

    """
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
        raise KeyError(f"Combination of subset and {value} not supported")
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
        return trial.suggest_float(key_name, *value, log=True)
    if key_type == "uniform" and _is_pair_of_floats(value):
        return trial.suggest_float(key_name, *value)
    raise KeyError(f"Combination of {key_type} and {value} not supported")


def _suggest_numeric_value(
    trial: optuna.trial.Trial, key_name: str, key_type: str, value: object
) -> float | int:
    if not isinstance(value, list) or len(value) < 2:
        raise KeyError(f"Combination of {key_type} and {value} not supported")
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
    raise KeyError(f"Combination of {key_type} and {value} not supported")


def _is_pair_of_floats(value: object) -> TypeGuard[list[float]]:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, float) for item in value)
    )
