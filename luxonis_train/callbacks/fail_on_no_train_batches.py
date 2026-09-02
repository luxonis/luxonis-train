from math import ceil

import lightning.pytorch as pl
from lightning.fabric.utilities.data import sized_len


class FailOnNoTrainBatches(pl.Callback):
    """Handles cases where number of training batches is 0 either due to
    too large effective batch size or skipping the last batch.
    """

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Ensure Lightning has computed the effective number of train batches.
        trainer.fit_loop.setup_data()
        if trainer.fit_loop.max_batches != 0:
            return
        raise RuntimeError(_no_train_batches_message(trainer, pl_module))


def _no_train_batches_message(
    trainer: pl.Trainer, pl_module: pl.LightningModule
) -> str:
    dataset_len, batch_size, drop_last = _loader_details(trainer)
    if batch_size is None:
        configured_batch_size = pl_module.cfg.trainer.batch_size  # type: ignore
        batch_size = (
            configured_batch_size
            if isinstance(configured_batch_size, int)
            else None
        )
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
    details: tuple[int | None, int | None, bool | None] = (None, None, None)
    for dataloader in dataloaders:
        if dataloader is None:
            continue
        details = _merge_loader_details(details, dataloader)
        if all(value is not None for value in details):
            break
    return details


def _merge_loader_details(
    details: tuple[int | None, int | None, bool | None], dataloader: object
) -> tuple[int | None, int | None, bool | None]:
    dataset_len, batch_size, drop_last = details
    if dataset_len is None:
        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None:
            dataset_len = sized_len(dataset)
    batch_size = (
        batch_size
        if batch_size is not None
        else getattr(dataloader, "batch_size", None)
    )
    drop_last = (
        drop_last
        if drop_last is not None
        else getattr(dataloader, "drop_last", None)
    )
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
    details = [
        f"dataset_size={dataset_len}" if dataset_len is not None else None,
        f"min_required_size={min_required}"
        if min_required is not None
        else None,
        (
            f"missing={min_required - dataset_len}"
            if dataset_len is not None
            and min_required is not None
            and dataset_len < min_required
            else None
        ),
    ]
    params = [
        f"batch_size={batch_size}" if batch_size is not None else None,
        f"world_size={world_size}",
        f"drop_last={drop_last}" if drop_last is not None else None,
        f"limit_train_batches={limit_batches}",
    ]
    detail_parts = [part for part in details if part is not None]
    params_msg = ", ".join(part for part in params if part is not None)
    return f"(details: {', '.join(detail_parts)}; params: {params_msg})"
