import pytorch_lightning as pl
import torch

from luxonis_train.strategies.triple_lr_sgd import TripleLRSGDStrategy


def test_triple_lr_sgd():
    class DummyModel(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.core = type("", (), {"loaders": {"train": range(10)}})()
            self.cfg = type(
                "",
                (),
                {"trainer": type("", (), {"batch_size": 1, "epochs": 50})()},
            )()
            self.linear = torch.nn.Linear(2, 1)
            self.lr_list = [[], [], []]

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x = batch
            y = self.forward(x)
            loss = torch.nn.functional.mse_loss(y, torch.zeros_like(y))
            return loss

        def configure_optimizers(self):
            self.strategy = TripleLRSGDStrategy(model)
            return self.strategy.configure_optimizers()

        def on_before_optimizer_step(self, optimizer):
            for i, param_group in enumerate(optimizer.param_groups):
                self.lr_list[i].append(param_group["lr"])

        def on_after_backward(self):
            self.strategy.update_parameters()

    model = DummyModel()

    dataset = torch.randn(model.core.loaders["train"].__len__(), 2)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
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
