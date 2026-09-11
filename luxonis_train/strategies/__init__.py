"""Training strategies that own the optimization schedule.

A strategy contributes parameter-group rules to the model-wide
partition, and can adjust its own groups on every step.
`TripleLRSGDStrategy` is the one this package ships: it runs SGD with a
warmup phase and optional cosine annealing.

Writing a custom strategy:
    Subclass `BaseTrainingStrategy` and implement:

    - ``rules()`` returns the ordered rules. Each rule carries a
      ``tag``, a structural ``selector`` over
      ``(module, module_name, parameter, parameter_name)``, an
      ``OptimizerConfig``, and an optional ``SchedulerConfig``. A rule
      that omits the scheduler inherits the base one.
    - ``get_base_configs()`` returns the base optimizer and scheduler
      pair. A node ``finetuning`` rule that names no optimizer
      inherits from it, and so does the default tail.
    - ``update_parameters()`` is optional. The trainer calls it after
      the backward pass and before the optimizers step. Reach a group
      through the handle that ``attach()`` passes for its rule tag:
      ``self.runtime.group(handle)["lr"] = ...``.

    The trainer evaluates the strategy rules after every node
    ``finetuning`` rule and before the default tail. A node override
    therefore wins, and every parameter the strategy does not claim
    still reaches an optimizer.

    Rules that share an optimizer name and a scheduler configuration
    collapse into one inner optimizer, with one parameter group for
    each rule. A single composite optimizer drives all inner
    optimizers, so training stays in the automatic optimization of
    Lightning.

The previous ``configure_optimizers()`` contract still works.
`LegacyStrategyAdapter` mounts such a strategy and logs a deprecation
warning. It stops working in the next minor release. To port a
strategy, express its parameter split as ``rules()``.
"""

from .base_strategy import BaseTrainingStrategy
from .legacy import LegacyStrategyAdapter
from .triple_lr_sgd import TripleLRSGDStrategy

__all__ = [
    "BaseTrainingStrategy",
    "LegacyStrategyAdapter",
    "TripleLRSGDStrategy",
]
