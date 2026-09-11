"""Ready-to-use configuration files shipped with the package.

Every command that accepts ``--config`` also accepts ``--model`` and an
optional ``--variant``, so you need no local file. For example,
``luxonis_train train --model detection --variant light`` loads
``detection_light_model.yaml``. The same works from Python with
``LuxonisModel(model="detection", variant="light")``.

``--variant`` accepts every variant the predefined model declares, not
only the variants with a file of their own. ``--model detection
--variant medium`` therefore works, although no
``detection_medium_model.yaml`` exists. Run ``luxonis_train
list-models`` to see the names.

``defaults.yaml`` holds the values every other file inherits.
``complex_model.yaml`` builds a graph by hand instead of naming a
predefined model. ``example_export.yaml`` and ``example_tuning.yaml``
show the ``exporter`` and ``tuner`` sections.

"""
