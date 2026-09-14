"""The construction of the Lightning trainer from the config."""

from typing import Any

import lightning.pytorch as pl

from luxonis_train.config import TrainerConfig


def create_trainer(cfg: TrainerConfig, **kwargs: Any) -> pl.Trainer:
    """Create a Lightning trainer from the ``trainer`` config section.

    The function passes these fields of ``cfg`` to the trainer:

    - ``accelerator``, ``devices``, ``strategy``, ``profiler``,
      ``deterministic``, ``gradient_clip_val``,
      ``gradient_clip_algorithm``, and ``overfit_batches`` under the
      same names;
    - ``epochs`` as ``max_epochs``;
    - ``validation_interval`` as ``check_val_every_n_epoch``;
    - ``n_sanity_val_steps`` as ``num_sanity_val_steps``.

    The other fields, such as ``precision``, do not reach the trainer
    through this function. The main trainer of `LuxonisModel` gets
    ``precision`` through ``kwargs``.

    Args:
        cfg (TrainerConfig): The ``trainer`` section of the config.
        **kwargs (``Any``): More keyword arguments for the trainer, such
            as ``logger``, ``callbacks``, or ``precision``. A key that
            the function already sets from ``cfg`` raises
            ``TypeError``.

    Returns:
        ``pl.Trainer``: The trainer.

    """
    return pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.epochs,
        check_val_every_n_epoch=cfg.validation_interval,
        num_sanity_val_steps=cfg.n_sanity_val_steps,
        profiler=cfg.profiler,
        deterministic=cfg.deterministic,
        gradient_clip_val=cfg.gradient_clip_val,
        gradient_clip_algorithm=cfg.gradient_clip_algorithm,
        overfit_batches=cfg.overfit_batches,
        **kwargs,
    )
