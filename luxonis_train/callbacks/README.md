# Callbacks

List of all supported callbacks.

## Table Of Contents

- [PytorchLightning Callbacks](#pytorchlightning-callbacks)
- [ExportOnTrainEnd](#exportontrainend)
- [LuxonisProgressBar](#luxonisprogressbar)
- [MetadataLogger](#metadatalogger)
- [TestOnTrainEnd](#testontrainend)
- [UploadCheckpoint](#uploadcheckpoint)

## PytorchLightning Callbacks

List of supported callbacks from `lightning.pytorch`.

- [GPUStatsMonitor](https://pytorch-lightning.readthedocs.io/en/1.5.10/api/pytorch_lightning.callbacks.gpu_stats_monitor.html)
- [DeviceStatsMonitor](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html#lightning.pytorch.callbacks.DeviceStatsMonitor)
- [EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping)
- [LearningRateMonitor](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#lightning.pytorch.callbacks.LearningRateMonitor)
- [ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint)
- [RichModelSummary](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html#lightning.pytorch.callbacks.RichModelSummary)

## ExportOnTrainEnd

Performs export on train end with best weights.

**Params**

| Key                  | Type                        | Default value | Description                                                                                                 |
| -------------------- | --------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| preferred_checkpoint | Literal\["metric", "loss"\] | metric        | Which checkpoint should we use. If preferred is not available then try to use the other one if its present. |

## LuxonisProgressBar

Custom rich text progress bar based on RichProgressBar from Pytorch Lightning.

## MetadataLogger

Callback that logs training metadata.

Metadata include all defined hyperparameters together with git hashes of `luxonis-ml` and `luxonis-train` packages. Also stores this information locally.

**Params**

| Key         | Type        | Default value | Description                                                                                                             |
| ----------- | ----------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| hyperparams | list\[str\] | \[\]          | List of hyperparameters to log. The hyperparameters are provided as config keys in dot notation. E.g. "trainer.epochs". |

## TestOnTrainEnd

Callback to perform a test run at the end of the training.

## UploadCheckpoint

Callback that uploads currently best checkpoint (based on validation loss) to the tracker location - where all other logs are stored.
