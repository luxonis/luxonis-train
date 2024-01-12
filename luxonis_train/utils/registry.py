"""This module implements a metaclass for automatic registration of classes."""


from luxonis_ml.utils.registry import Registry

CALLBACKS = Registry(name="callbacks")
"""Registry for all callbacks."""

LOADERS = Registry(name="loaders")
"""Registry for all loaders."""

LOSSES = Registry(name="losses")
"""Registry for all losses."""

METRICS = Registry(name="metrics")
"""Registry for all metrics."""

MODELS = Registry(name="models")
"""Registry for all models."""

NODES = Registry(name="nodes")
"""Registry for all nodes."""

OPTIMIZERS = Registry(name="optimizers")
"""Registry for all optimizers."""

SCHEDULERS = Registry(name="schedulers")
"""Registry for all schedulers."""

VISUALIZERS = Registry(name="visualizers")
"""Registry for all visualizers."""
