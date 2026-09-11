"""Helpers behind the methods of `LuxonisModel`.

Each module here carries the part of a command that does not belong in
the class itself: the ONNX export, the NN Archive, the AIMET
quantization, the inference loop, the pre-annotation pass, the trainer
construction, and the Optuna study.

"""
