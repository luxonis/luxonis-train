"""Training strategies that own the optimization schedule.

A strategy adds parameter-group rules to the partition of the model
parameters, and can change its own groups on every step. The package
ships `TripleLRSGDStrategy`. It runs SGD with a warmup and a cosine
or linear decay of the learning rate.

Writing a custom strategy:
    Subclass `BaseTrainingStrategy` and implement these methods:

    - ``rules()`` returns the rules in order. Each `StrategyRule` has a
      ``tag``, a ``selector`` over
      ``(module, module_name, parameter, parameter_name)``, an
      ``OptimizerConfig``, and an optional ``SchedulerConfig``. A rule
      without a scheduler uses the base scheduler.
    - ``get_base_configs()`` returns the base optimizer config and the
      base scheduler config. The ``finetuning`` entries of the nodes
      merge their overrides into them, and the default rule uses them.
    - ``update_parameters()`` is optional. `TrainingManager` calls it
      after each backward pass, before the optimizers step. Reach a
      group through a handle that ``attach()`` stores for the rule tag:
      ``self.runtime.group(handle)["lr"] = ...``.

    `resolve_training_plan` evaluates the strategy rules after the
    ``finetuning`` entries of every node and before the default rule. A
    node entry therefore wins, and every parameter that the strategy
    does not claim still gets an optimizer.

    Rules with the same optimizer name and the same scheduler config
    share one inner optimizer, with one parameter group for each rule.
    With more than one inner optimizer, one `CompositeOptimizer` drives
    them, so the training stays in the automatic optimization of
    Lightning.

A strategy of the previous ``configure_optimizers()`` API still works
until the next minor release.
`luxonis_train.lightning.utils.build_training_strategy` logs a
deprecation warning and wraps such a strategy in
`LegacyStrategyAdapter`. To port a strategy, express its parameter
split as ``rules()``, and move its per-step logic to
``update_parameters()``.

"""

from .base_strategy import BaseTrainingStrategy
from .legacy import LegacyStrategyAdapter
from .triple_lr_sgd import TripleLRSGDStrategy

__all__ = [
    "BaseTrainingStrategy",
    "LegacyStrategyAdapter",
    "TripleLRSGDStrategy",
]
