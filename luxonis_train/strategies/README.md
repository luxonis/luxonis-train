# Training Strategies

Training strategies define how the optimization process is orchestrated during training. Each strategy combines specific configurations and update mechanisms to achieve optimal convergence for various tasks. Strategies are flexible and customizable based on the needs of the training pipeline.

## Table Of Contents

- [`TripleLRSGDStrategy`](#triplelrsgdstrategy)

## `TripleLRSGDStrategy`

The `TripleLRSGDStrategy` implements a staged learning rate schedule with warmup and momentum adjustments, tailored for stable and efficient convergence.

**Parameters:**

| Key                | Type    | Default Value | Description                                                |
| ------------------ | ------- | ------------- | ---------------------------------------------------------- |
| `warmup_epochs`    | `int`   | `3`           | Number of epochs for the learning rate warmup phase.       |
| `warmup_bias_lr`   | `float` | `0.1`         | Learning rate for bias parameters during the warmup phase. |
| `warmup_momentum`  | `float` | `0.8`         | Momentum used during the warmup phase.                     |
| `lr`               | `float` | `0.02`        | Base learning rate for the main training phase.            |
| `lre`              | `float` | `0.0002`      | Ending learning rate after the scheduled decay.            |
| `momentum`         | `float` | `0.937`       | Momentum factor for stable optimization.                   |
| `weight_decay`     | `float` | `0.0005`      | Weight decay (L2 penalty) for regularization.              |
| `nesterov`         | `bool`  | `True`        | Whether to use Nesterov momentum during optimization.      |
| `cosine_annealing` | `bool`  | `True`        | Whether to use cosine annealing.                           |

## Writing A Custom Strategy

A strategy contributes **parameter-group rules** to the model-wide
partition and may adjust its groups every step. Subclass
`BaseTrainingStrategy` and implement:

- `rules() -> list[StrategyRule]` — ordered rules, each with a `tag`, a
  structural `selector` (a callable over
  `(module, module_name, parameter, parameter_name)`), an
  `OptimizerConfig`, and optionally a `SchedulerConfig` (omitted rules
  inherit the strategy's base scheduler). Rules are evaluated after
  every node-level `finetuning` rule and before the default tail, so
  node overrides win and every parameter the strategy does not claim
  still ends up in an optimizer.
- `get_base_configs() -> (OptimizerConfig, SchedulerConfig)` — the base
  pair. Node `finetuning` rules that omit names inherit from it, and
  the default tail uses it.
- optionally `update_parameters()` — a per-step hook (called after the
  backward pass, before the optimizers step). Use the group handles
  passed to `attach()` (keyed by rule tag) to reach your groups:
  `self.runtime.group(handle)["lr"] = ...`.

Rules sharing an optimizer name and scheduler configuration collapse
into one inner optimizer with one parameter group per rule; all inner
optimizers are driven through a single composite optimizer under
Lightning's automatic optimization.

### Migrating From The Deprecated API

Strategies implementing the previous
`configure_optimizers()`/`update_parameters()` contract still work: they
are mounted in compatibility mode (their optimizer becomes an opaque
inner of the composite) with a deprecation warning, and will stop
working in the next minor release. To port a strategy, express its
parameter split as `rules()` and move any per-step logic onto
`update_parameters()` with group handles.
