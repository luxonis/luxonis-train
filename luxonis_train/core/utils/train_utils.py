import lightning.pytorch as pl

from luxonis_train.utils.config import Config


def create_trainer(cfg: Config, **kwargs) -> pl.Trainer:
    """Creates Pytorch Lightning trainer.

    @type cfg: Config
    @param cfg: Configuration object.
    @param kwargs: Additional arguments to pass to the trainer.
    @rtype: pl.Trainer
    @return: Pytorch Lightning trainer.
    """
    return pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.trainer.validation_interval,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        profiler=cfg.trainer.profiler,
        **kwargs,
    )
