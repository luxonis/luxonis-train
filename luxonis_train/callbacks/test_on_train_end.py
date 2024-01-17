import lightning.pytorch as pl
from luxonis_ml.data import LuxonisDataset, ValAugmentations
from torch.utils.data import DataLoader

from luxonis_train.utils.config import Config
from luxonis_train.utils.loaders import LuxonisLoaderTorch, collate_fn
from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class TestOnTrainEnd(pl.Callback):
    """Callback to perform a test run at the end of the training."""

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        cfg: Config = pl_module.cfg

        dataset = LuxonisDataset(
            dataset_name=cfg.dataset.name,
            team_id=cfg.dataset.team_id,
            dataset_id=cfg.dataset.id,
            bucket_type=cfg.dataset.bucket_type,
            bucket_storage=cfg.dataset.bucket_storage,
        )

        loader_test = LuxonisLoaderTorch(
            dataset,
            view=cfg.dataset.test_view,
            augmentations=ValAugmentations(
                image_size=cfg.trainer.preprocessing.train_image_size,
                augmentations=[
                    i.model_dump() for i in cfg.trainer.preprocessing.augmentations
                ],
                train_rgb=cfg.trainer.preprocessing.train_rgb,
                keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
            ),
        )
        pytorch_loader_test = DataLoader(
            loader_test,
            batch_size=cfg.trainer.batch_size,
            num_workers=cfg.trainer.num_workers,
            collate_fn=collate_fn,
        )
        trainer.test(pl_module, pytorch_loader_test)
