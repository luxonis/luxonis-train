import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from luxonis_train.strategies.triple_lr_sgd import TripleLRSGDStrategy


class _Core:
    loaders = {"train": range(10)}


class _Cfg:
    class Trainer:
        batch_size = 1
        epochs = 50

    trainer = Trainer()


def test_triple_lr_sgd():
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.core = _Core()
            self.cfg = _Cfg()
            self.linear = torch.nn.Linear(2, 1)
            self.lr_list = [[], [], []]

        def forward(self, x: Tensor) -> Tensor:
            return self.linear(x)

        def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
            x = batch
            y = self.forward(x)
            return torch.nn.functional.mse_loss(y, torch.zeros_like(y))

        def configure_optimizers(
            self,
        ) -> tuple[list[Optimizer], list[LambdaLR]]:
            self.strategy = TripleLRSGDStrategy(model)  # type: ignore
            return self.strategy.configure_optimizers()

        def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
            for i, param_group in enumerate(optimizer.param_groups):
                self.lr_list[i].append(param_group["lr"])

        def on_after_backward(self) -> None:
            self.strategy.update_parameters()

    model = DummyModel()

    dataset = torch.randn(model.core.loaders["train"].__len__(), 2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)  # type: ignore
    trainer = pl.Trainer(max_epochs=model.cfg.trainer.epochs)
    trainer.fit(model, dataloader)

    cases = [
        (0, 0, 0.0, None),
        (1, 0, 0.0, None),
        (2, 0, 0.1, None),
        (0, 100, 0.018, 3e-4),
        (1, 100, 0.018, 3e-4),
        (2, 100, 0.018, 3e-4),
        (0, -1, 0.0002, 5e-5),
        (1, -1, 0.0002, 5e-5),
        (2, -1, 0.0002, 5e-5),
        (0, 50, 0.0098, 1e-4),
        (1, 50, 0.0098, 1e-4),
        (2, 50, 0.0597, 1e-4),
        (0, 150, 0.0159, 3e-4),
        (1, 150, 0.0159, 3e-4),
        (2, 150, 0.0159, 3e-4),
    ]

    for group_idx, step, expected, tol in cases:
        value = model.lr_list[group_idx][step]
        if tol is None:
            assert value == expected
        else:
            assert abs(value - expected) < tol
