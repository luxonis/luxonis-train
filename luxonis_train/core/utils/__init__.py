"""Helpers that the methods of `LuxonisModel` call.

The modules keep the steps of the commands out of the class:

- `aimet_utils`: the AIMET quantization of `LuxonisModel.quantize`.
- `annotate_utils`: the pre-annotation of `LuxonisModel.annotate`.
- `archive_utils`: the inputs, the outputs, and the heads of the NN
  Archive.
- `export_utils`: the ONNX export, and the conversions that follow it.
- `infer_utils`: the inference loops of `LuxonisModel.infer`.
- `train_utils`: the construction of the Lightning trainer.
- `tune_utils`: the trial parameters of `LuxonisModel.tune`.

"""
