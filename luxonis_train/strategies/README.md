# Training Strategies

Training strategies define how the optimization process is orchestrated during training. Each strategy combines specific configurations and update mechanisms to achieve optimal convergence for various tasks. Strategies are flexible and customizable based on the needs of the training pipeline.

## Table Of Contents

- [`TripleLRSGDStrategy`](#triplelrsgdstrategy)

## `TripleLRSGDStrategy`

The `TripleLRSGDStrategy` implements a staged learning rate schedule with warmup and momentum adjustments, tailored for stable and efficient convergence.

**Parameters:**

| Key               | Type    | Default Value | Description                                                |
| ----------------- | ------- | ------------- | ---------------------------------------------------------- |
| `warmup_epochs`   | `int`   | `3`           | Number of epochs for the learning rate warmup phase.       |
| `warmup_bias_lr`  | `float` | `0.1`         | Learning rate for bias parameters during the warmup phase. |
| `warmup_momentum` | `float` | `0.8`         | Momentum used during the warmup phase.                     |
| `lr`              | `float` | `0.02`        | Base learning rate for the main training phase.            |
| `lre`             | `float` | `0.0002`      | Ending learning rate after the scheduled decay.            |
| `momentum`        | `float` | `0.937`       | Momentum factor for stable optimization.                   |
| `weight_decay`    | `float` | `0.0005`      | Weight decay (L2 penalty) for regularization.              |
| `nesterov`        | `bool`  | `True`        | Whether to use Nesterov momentum during optimization.      |
