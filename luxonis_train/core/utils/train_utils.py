from typing import Any

import lightning.pytorch as pl

# from lightning.pytorch.plugins.environments import LightningEnvironment
# from lightning.pytorch.strategies import DDPStrategy
from luxonis_train.config import TrainerConfig


def create_trainer(cfg: TrainerConfig, **kwargs: Any) -> pl.Trainer:
    """Creates Pytorch Lightning trainer.

    @type cfg: Config
    @param cfg: Configuration object.
    @param kwargs: Additional arguments to pass to the trainer.
    @rtype: pl.Trainer
    @return: Pytorch Lightning trainer.
    """
    # os.environ["NCCL_DEBUG"] = "INFO"
    # os.environ["NCCL_SOCKET_IFNAME"] = (
    #     "eth0"  # Adjust based on your network interface
    # )
    # os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand if not available

    # strategy = (  # WARNING: DDP training FREEZES when using LightningEnvironment!!!
    #     DDPStrategy(cluster_environment=LightningEnvironment())
    #     if cfg.strategy.lower() == "ddp"
    #     else cfg.strategy
    # )
    # printenv | grep -E "WORLD_SIZE|RANK|LOCAL_RANK|MASTER_ADDR|MASTER_PORT"

    # output:
    # root@detection-zz22crkgtlnxzc-luxonistrain-0-0:/workspace# printenv | grep -E "WORLD_SIZE|RANK|LOCAL_RANK|MASTER_ADDR|MASTER_PORT"
    # TORCHX_RANK0_HOST=localhost
    # WORLD_SIZE=2

    # MANUALY EXPORT:
    # export NCCL_DEBUG=INFO
    # export NCCL_SOCKET_IFNAME=eth0  # Use the correct network interface
    # export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

    # START TRAINGING luxonis_trian train --config TEST_GCP.yaml
    # Manualy trained -> WORKED!!!

    return pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        strategy=cfg.strategy,
        max_epochs=cfg.epochs,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.validation_interval,
        num_sanity_val_steps=cfg.n_sanity_val_steps,
        profiler=cfg.profiler,
        deterministic=cfg.deterministic,
        **kwargs,
    )
