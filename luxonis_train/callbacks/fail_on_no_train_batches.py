from math import ceil

import lightning.pytorch as pl
from lightning.fabric.utilities.data import sized_len


class FailOnNoTrainBatches(pl.Callback):
    """Handles cases where number of training batches is 0 either due to
    too large effective batch size or skipping the last batch."""

    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Ensure Lightning has computed the effective number of train batches.
        trainer.fit_loop.setup_data()
        if trainer.fit_loop.max_batches == 0:
            dataset_len = None
            batch_size = None
            drop_last = None

            combined_loader = trainer.fit_loop._combined_loader
            flattened = getattr(combined_loader, "flattened", None)
            dataloaders = (
                flattened if isinstance(flattened, list) else [combined_loader]
            )
            # try to get info from dataloader directly
            dataloaders = [dl for dl in dataloaders if dl is not None]
            for dl in dataloaders:
                if dataset_len is None:
                    dataset = getattr(dl, "dataset", None)
                    if dataset is not None:
                        dataset_len = sized_len(dataset)
                if batch_size is None:
                    batch_size = getattr(dl, "batch_size", None)
                if drop_last is None:
                    drop_last = getattr(dl, "drop_last", None)
                if (
                    dataset_len is not None
                    and batch_size is not None
                    and drop_last is not None
                ):
                    break

            # fallback to config
            if batch_size is None:
                batch_size = pl_module.cfg.trainer.batch_size  # type: ignore

            world_size = trainer.world_size
            limit_batches = trainer.limit_train_batches

            min_required = None
            min_batches_needed = None

            # check if we are limiting number of batches
            if isinstance(limit_batches, int):
                if limit_batches > 0:
                    min_batches_needed = 1
            elif isinstance(limit_batches, float) and limit_batches > 0.0:
                min_batches_needed = ceil(1.0 / limit_batches)

            if (
                batch_size is not None
                and drop_last is not None
                and min_batches_needed is not None
            ):
                if drop_last:
                    min_required = batch_size * world_size * min_batches_needed
                else:
                    min_required = (
                        min_batches_needed - 1
                    ) * batch_size * world_size + 1

            detail_parts: list[str] = []
            if dataset_len is not None:
                detail_parts.append(f"dataset_size={dataset_len}")
            if min_required is not None:
                detail_parts.append(f"min_required_size={min_required}")
            if (
                dataset_len is not None
                and min_required is not None
                and dataset_len < min_required
            ):
                detail_parts.append(f"missing={min_required - dataset_len}")

            params = [
                f"batch_size={batch_size}" if batch_size is not None else None,
                f"world_size={world_size}",
                f"drop_last={drop_last}" if drop_last is not None else None,
                f"limit_train_batches={limit_batches}",
            ]
            params_msg = ", ".join(p for p in params if p is not None)
            detail_msg = (
                f" (details: {', '.join(detail_parts)}; params: {params_msg})"
                if detail_parts or params_msg
                else ""
            )

            raise RuntimeError(
                "No training batches found. "
                "Your dataset is smaller than the effective batch size "
                "or skip_last_batch=True removed the last batch. "
                f"{detail_msg}"
            )
