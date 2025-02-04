from typing import Any

import lightning.pytorch as pl

from luxonis_train.config import TrainerConfig


def create_trainer(cfg: TrainerConfig, **kwargs: Any) -> pl.Trainer:
    """Creates Pytorch Lightning trainer.

    @type cfg: Config
    @param cfg: Configuration object.
    @param kwargs: Additional arguments to pass to the trainer.
    @rtype: pl.Trainer
    @return: Pytorch Lightning trainer.
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
        **kwargs,
    )
