from math import ceil

import lightning.pytorch as pl
from lightning.fabric.utilities.data import sized_len

import luxonis_train as lxt


class FailOnNoTrainBatches(pl.Callback):
    """Handles cases where number of training batches is 0 either due to
    too large effective batch size or skipping the last batch.
    """

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        # Ensure Lightning has computed the effective number of train batches.
        trainer.fit_loop.setup_data()
        if trainer.fit_loop.max_batches != 0:
            return
        raise RuntimeError(_no_train_batches_message(trainer, pl_module))


def _no_train_batches_message(
    trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
) -> str:
    dataset_len, batch_size, drop_last = _loader_details(trainer)
    if batch_size is None:
        batch_size = pl_module.cfg.trainer.batch_size
    min_required = _minimum_required_size(
        batch_size,
        drop_last,
        trainer.world_size,
        trainer.limit_train_batches,
    )
    detail_msg = _format_details(
        dataset_len,
        min_required,
        batch_size,
        trainer.world_size,
        drop_last,
        trainer.limit_train_batches,
    )
    return (
        "No training batches found. Your dataset is smaller than the effective "
        "batch size or skip_last_batch=True removed the last batch. "
        f"{detail_msg}"
    )


def _loader_details(
    trainer: pl.Trainer,
) -> tuple[int | None, int | None, bool | None]:
    combined_loader = trainer.fit_loop._combined_loader
    flattened = getattr(combined_loader, "flattened", None)
    dataloaders = (
        flattened if isinstance(flattened, list) else [combined_loader]
    )
    dataloaders = [loader for loader in dataloaders if loader is not None]
    dataset_len = batch_size = drop_last = None
    for dataloader in dataloaders:
        if dataset_len is None:
            dataset = getattr(dataloader, "dataset", None)
            if dataset is not None:
                dataset_len = sized_len(dataset)
        if batch_size is None:
            batch_size = getattr(dataloader, "batch_size", None)
        if drop_last is None:
            drop_last = getattr(dataloader, "drop_last", None)
        if None not in (dataset_len, batch_size, drop_last):
            break
    return dataset_len, batch_size, drop_last


def _minimum_required_size(
    batch_size: int | None,
    drop_last: bool | None,
    world_size: int,
    limit_batches: float,
) -> int | None:
    min_batches_needed = _minimum_batch_count(limit_batches)
    if batch_size is None or drop_last is None or min_batches_needed is None:
        return None
    if drop_last:
        return batch_size * world_size * min_batches_needed
    return (min_batches_needed - 1) * batch_size * world_size + 1


def _minimum_batch_count(limit_batches: float) -> int | None:
    if isinstance(limit_batches, int):
        return 1 if limit_batches > 0 else None
    return ceil(1.0 / limit_batches) if limit_batches > 0.0 else None


def _format_details(
    dataset_len: int | None,
    min_required: int | None,
    batch_size: int | None,
    world_size: int,
    drop_last: bool | None,
    limit_batches: float,
) -> str:
    missing = None
    if (
        dataset_len is not None
        and min_required is not None
        and dataset_len < min_required
    ):
        missing = min_required - dataset_len

    details = _format_fields(
        dataset_size=dataset_len,
        min_required_size=min_required,
        missing=missing,
    )
    params = _format_fields(
        batch_size=batch_size,
        world_size=world_size,
        drop_last=drop_last,
        limit_train_batches=limit_batches,
    )
    return f"(details: {details}; params: {params})"


def _format_fields(**parts: object) -> str:
    return ", ".join(
        f"{name}={value}" for name, value in parts.items() if value is not None
    )
