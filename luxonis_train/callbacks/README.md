# Callbacks

List of all supported callbacks.

## Table Of Contents

- [PytorchLightning Callbacks](#pytorchlightning-callbacks)
- [ExportOnTrainEnd](#exportontrainend)
- [LuxonisProgressBar](#luxonisprogressbar)
- [MetadataLogger](#metadatalogger)
- [TestOnTrainEnd](#testontrainend)

## PytorchLightning Callbacks

List of supported callbacks from `lightning.pytorch`.

- [DeviceStatsMonitor](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html#lightning.pytorch.callbacks.DeviceStatsMonitor)
- [ EarlyStopping ](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping)
- [ LearningRateMonitor ](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.LearningRateMonitor.html#lightning.pytorch.callbacks.LearningRateMonitor)
- [ ModelCheckpoint ](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint)
- [ RichModelSummary ](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html#lightning.pytorch.callbacks.RichModelSummary)
  - Added automatically if `use_rich_text` is set to `True` in [config](../../../configs/README.md#topleveloptions).

## ExportOnTrainEnd

Performs export on train end with best weights according to the validation loss.

**Params**

| Key              | Type | Default value | Description                                                                            |
| ---------------- | ---- | ------------- | -------------------------------------------------------------------------------------- |
| upload_to_mlflow | bool | False         | If set to True, overrides the upload url in exporter with currently active MLFlow run. |

## LuxonisProgressBar

Custom rich text progress bar based on RichProgressBar from Pytorch Lightning.
Added automatically if `use_rich_text` is set to `True` in [config](../../../configs/README.md#topleveloptions).

## MetadataLogger

Callback that logs training metadata.

Metadata include all defined hyperparameters together with git hashes of `luxonis-ml` and `luxonis-train` packages. Also stores this information locally.

**Params**

| Key         | Type        | Default value | Description                                                                                                             |
| ----------- | ----------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| hyperparams | list\[str\] | \[\]          | List of hyperparameters to log. The hyperparameters are provided as config keys in dot notation. E.g. "trainer.epochs". |

## TestOnTrainEnd

Callback to perform a test run at the end of the training.
