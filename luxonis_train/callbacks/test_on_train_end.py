import lightning.pytorch as pl
from luxonis_ml.data import LuxonisDataset, ValAugmentations
from torch.utils.data import DataLoader

from luxonis_train.utils.loaders import LuxonisLoaderTorch, collate_fn
from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class TestOnTrainEnd(pl.Callback):
    """Callback to perform a test run at the end of the training."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dataset = LuxonisDataset(
            dataset_name=pl_module.cfg.dataset.dataset_name,
            team_id=pl_module.cfg.dataset.team_id,
            dataset_id=pl_module.cfg.dataset.dataset_id,
            bucket_type=pl_module.cfg.dataset.bucket_type,
            bucket_storage=pl_module.cfg.dataset.bucket_storage,
        )

        loader_test = LuxonisLoaderTorch(
            dataset,
            view=pl_module.cfg.dataset.test_view,
            augmentations=ValAugmentations(
                image_size=pl_module.cfg.trainer.preprocessing.train_image_size,
                augmentations=[
                    i.model_dump()
                    for i in pl_module.cfg.trainer.preprocessing.augmentations
                ],
                train_rgb=pl_module.cfg.trainer.preprocessing.train_rgb,
                keep_aspect_ratio=pl_module.cfg.trainer.preprocessing.keep_aspect_ratio,
            ),
        )
        pytorch_loader_test = DataLoader(
            loader_test,
            batch_size=pl_module.cfg.trainer.batch_size,
            num_workers=pl_module.cfg.trainer.num_workers,
            collate_fn=collate_fn,
        )
        trainer.test(pl_module, pytorch_loader_test)
