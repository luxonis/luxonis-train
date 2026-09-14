"""Ready-to-use config files shipped with the package.

The ``train``, ``tune``, ``inspect``, ``test``, ``infer``, ``annotate``,
``export``, ``archive``, ``convert``, and ``quantize`` commands accept
``--model`` and an optional ``--variant`` instead of ``--config``. A run
then needs no local config file. For example,
``luxonis_train train --model detection --variant light`` loads
``detection_light_model.yaml``. The Python equivalent is
``LuxonisModel(model="detection", variant="light")``.
`luxonis_train.config.predefined` finds the file.

``--variant`` accepts every variant that the predefined model declares,
not only the variants with a file of their own. For example, ``--model
detection --variant medium`` loads ``detection_light_model.yaml`` and
selects the ``medium`` variant, because no
``detection_medium_model.yaml`` exists. Run ``luxonis_train
list-models`` to see the model names and their variants.

The other files are examples:

- ``defaults.yaml`` shows the default values of the config fields. The
  package does not load it.
- ``complex_model.yaml`` builds a node graph by hand instead of naming
  a predefined model.
- ``example_export.yaml`` and ``example_tuning.yaml`` show the
  ``exporter`` and the ``tuner`` sections.

"""
