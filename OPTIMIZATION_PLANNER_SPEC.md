# Optimization Planner Spec

Status: rough draft v0

## Summary

Optimizer and scheduler definition should have one owner. The current feature
splits that responsibility across the Lightning module, `Nodes`, training
strategies, and `TrainingManager`. That makes it difficult to reason about
which parameters are optimized, which scheduler owns each optimizer, how
freezing and unfreezing should work, and how node-level finetuning should
compose with training strategies.

This spec proposes a central optimization-planning layer. The planner builds a
single optimization plan from model nodes, base trainer config, node finetuning
config, freezing config, and optional training strategy behavior. The
Lightning module materializes this plan in `configure_optimizers()`, while
`TrainingManager` consumes the same plan metadata for lifecycle hooks.

## Problem

The parameter-groups feature needs to support three overlapping use cases:

1. Standard training with one optimizer and one scheduler.
2. Node-level finetuning with custom parameter groups, optimizer params, and
   scheduler params.
3. Training strategies such as `TripleLRSGDStrategy`, with further finetuning
   layered on top.

The implementation currently makes these responsibilities ambiguous:

- `configure_optimizers()` can combine strategy-created optimizers with
  finetuning-created optimizers.
- `configure_callbacks()` calls `configure_optimizers()` just to count
  optimizers, which constructs optimizers more than once.
- Parameter ownership can be duplicated when a strategy optimizer already owns
  parameters and finetuning then builds additional optimizers over the same
  parameters.
- Scheduler stepping semantics diverge from Lightning's normal monitor-based
  behavior when multiple optimizers are used.
- Strategy-specific assumptions, such as `TripleLRSGD` bias group index `2`,
  are fragile once finetuning creates extra groups.

## Goals

- Define optimizer and scheduler ownership in one place.
- Ensure every trainable parameter is assigned to at most one optimizer unless
  an explicit strategy opts into overlapping ownership.
- Let finetuning override optimizer parameters on top of a training strategy.
- Keep training strategies responsible for strategy-specific grouping and
  dynamic updates.
- Keep `TrainingManager` responsible for training lifecycle events, not primary
  optimizer construction.
- Preserve Lightning compatibility by returning optimizers and schedulers from
  `LightningModule.configure_optimizers()`.
- Make unsupported combinations fail with clear errors instead of silently
  creating ambiguous optimization behavior.

## Non-Goals

- Do not introduce arbitrary overlapping optimizers by default.
- Do not make every scheduler mode work with manual optimization immediately.
- Do not make `TrainingManager` the only place that constructs optimizers,
  because Lightning expects optimizer construction through
  `configure_optimizers()`.
- Do not preserve current behavior if current behavior double-optimizes a
  parameter or silently drops scheduler parameters.

## Proposed Ownership

### `OptimizationPlanner`

New central owner for building an optimization plan.

Responsibilities:

- Discover parameter candidates from model nodes.
- Attach metadata to each parameter.
- Apply node-level finetuning selectors.
- Merge base trainer optimizer and scheduler config with finetuning overrides.
- Apply freezing constraints.
- Delegate strategy-specific transformation to the configured training
  strategy.
- Materialize optimizers and schedulers exactly once.
- Return metadata needed by `TrainingManager`.

### `LuxonisLightningModule`

Responsibilities:

- Own the `OptimizationPlanner` instance.
- Call the planner from `configure_optimizers()`.
- Return the planner's optimizers and schedulers to Lightning.
- Avoid calling `configure_optimizers()` from `configure_callbacks()`.

### `TrainingManager`

Responsibilities:

- Freeze nodes before training.
- Unfreeze parameter groups according to planner metadata.
- Call training strategy lifecycle hooks.
- Coordinate manual optimization only if the finalized plan requires it.

`TrainingManager` should not independently build optimizers. It should consume
the planner output.

### Training Strategies

Responsibilities:

- Transform parameter group specs into strategy-specific group specs.
- Define strategy-specific optimizer defaults.
- Define strategy-specific scheduler behavior.
- Define dynamic lifecycle hooks such as warmup updates.
- Declare which kinds of finetuning overrides are supported.

Strategies should not directly compete with the planner for parameter
ownership.

## Core Data Model

### `ParameterSpec`

Internal representation of one model parameter before grouping.

Suggested fields:

```python
@dataclass
class ParameterSpec:
    name: str
    parameter: nn.Parameter
    node_name: str
    formatted_node_name: str
    module_name: str
    module_type: str
    parameter_name: str
    role: Literal["weight", "bias", "batch_norm_weight", "other"]
    requires_grad: bool
    frozen_until_epoch: int | None
```

The planner should match finetuning selectors against this metadata, not
against ad hoc strings only.

### `ParameterGroupSpec`

Internal representation of a future optimizer parameter group.

Suggested fields:

```python
@dataclass
class ParameterGroupSpec:
    params: list[nn.Parameter]
    optimizer_name: str
    optimizer_params: dict[str, Any]
    scheduler_name: str
    scheduler_params: dict[str, Any]
    source: str
    tags: set[str]
    strategy_role: str | None = None
```

`source` is useful for logging and debugging, for example
`trainer.optimizer`, `node:Backbone.finetuning[0]`, or
`strategy:TripleLRSGD.bias`.

### `OptimizationPlan`

Final planner output.

Suggested fields:

```python
@dataclass
class OptimizationPlan:
    optimizers: list[Optimizer]
    schedulers: list[LRScheduler | LRSchedulerConfig]
    parameter_groups: list[ParameterGroupSpec]
    parameter_to_optimizer: dict[int, int]
    frozen_specs: list[ParameterSpec]
    uses_manual_optimization: bool
    strategy: BaseTrainingStrategy | None
```

## Planning Algorithm

1. Collect all trainable and frozen parameters from `Nodes`.
2. Build a `ParameterSpec` for each parameter.
3. Apply freezing:
   - Frozen parameters start with `requires_grad=False`.
   - The plan records when each parameter can be unfrozen.
4. Apply base trainer optimizer and scheduler config to all eligible
   parameters.
5. Apply node-level finetuning overlays in config order.
6. Validate ownership:
   - A parameter can belong to one finalized optimizer by default.
   - Overlapping ownership is allowed only if a strategy explicitly supports it.
7. Normalize groups to keep optimizer instances low:
   - Merge parameters with identical optimizer arguments and scheduler
     arguments into the same parameter group.
   - Prefer one optimizer instance per optimizer class and compatible scheduler
     config.
   - Use multiple parameter groups inside one optimizer when optimizer class is
     the same but group options differ.
   - Use multiple optimizer instances only when optimizer class or scheduler
     semantics require it.
8. If a training strategy is configured, pass the group specs into the
   strategy for transformation.
9. Validate strategy output.
10. Materialize optimizers.
11. Materialize schedulers.
12. Return one `OptimizationPlan`.

## Finetuning Semantics

Finetuning should be treated as overlays on parameter group specs.

Example:

```yaml
model:
  nodes:
    - name: EfficientRep
      finetuning:
        optimizer:
          params:
            lr: 0.002

    - name: OCRCTCHead
      finetuning:
        - parameters:
            - module_type: Linear
          optimizer:
            params:
              weight_decay: 0.0004
```

Rules:

- If `parameters` is omitted, the finetuning block applies to all parameters
  in the node.
- If `optimizer.name` is omitted, it inherits the active base optimizer name.
- If `scheduler.name` is omitted, it inherits the active base scheduler name.
- `optimizer.params` is merged into inherited optimizer params.
- `scheduler.params` is merged into inherited scheduler params.
- Multiple optimizer classes are allowed when no training strategy is active.
- The planner should keep the number of optimizer instances as low as possible.
  Parameters with the same optimizer and scheduler arguments should be merged
  into the same optimizer group, and compatible groups should share the same
  optimizer instance.

Selector precedence:

1. More specific selectors override less specific selectors.
2. Ties are resolved by config order, later wins.
3. A parameter still ends up in exactly one final group unless multi-optimizer
   mode is explicitly enabled.

Specificity should be deterministic. A rough scoring model:

- Exact parameter path match is most specific.
- `module_type` plus `name` is more specific than either one alone.
- `name` is more specific than `module_type`.
- Omitted `parameters` is least specific.

## Training Strategy Integration

Training strategies should transform parameter groups instead of creating
independent optimizers over the model.

Proposed strategy API:

```python
class BaseTrainingStrategy:
    def optimizer_defaults(self) -> OptimizerConfig: ...

    def scheduler_defaults(self) -> SchedulerConfig: ...

    def transform_groups(
        self,
        groups: list[ParameterGroupSpec],
    ) -> list[ParameterGroupSpec]: ...

    def create_scheduler(
        self,
        optimizer: Optimizer,
        group_specs: list[ParameterGroupSpec],
    ) -> LRScheduler | LRSchedulerConfig: ...

    def supports_optimizer_override(self, optimizer_name: str) -> bool: ...

    def supports_multi_optimizer(self) -> bool:
        return False

    def on_after_backward(self, plan: OptimizationPlan) -> None: ...
```

The exact API can be smaller, but the strategy needs three capabilities:

- Provide defaults.
- Transform groups.
- Declare unsupported overrides.

## `TripleLRSGDStrategy`

`TripleLRSGDStrategy` should become a group transformer.

Current behavior:

- Creates one SGD optimizer.
- Splits parameters into BatchNorm weights, regular weights, and biases.
- Applies weight decay only to regular weights.
- Applies special warmup LR to the bias group.
- Assumes the bias group is index `2`.

Proposed behavior:

1. Planner builds base groups from nodes and finetuning.
2. `TripleLRSGDStrategy.transform_groups()` splits each compatible group into:
   - BatchNorm weights.
   - Regular weights.
   - Biases.
3. Finetuning overrides are preserved on the resulting split groups.
4. Each group gets a stable role tag:
   - `strategy_role="batch_norm_weight"`
   - `strategy_role="regular_weight"`
   - `strategy_role="bias"`
5. Warmup uses the role tag, not the group index.

Example:

```yaml
trainer:
  training_strategy:
    name: TripleLRSGDStrategy
    params:
      lr: 0.02
      weight_decay: 0.0005

model:
  nodes:
    - name: EfficientRep
      finetuning:
        optimizer:
          params:
            lr: 0.002

    - name: EfficientBBoxHead
      finetuning:
        - parameters:
            - module_type: Conv2d
          optimizer:
            params:
              weight_decay: 0.0001
```

Expected interpretation:

- `TripleLRSGDStrategy` remains the optimizer strategy.
- Backbone groups inherit SGD and strategy scheduler behavior.
- Backbone groups use `lr=0.002`.
- Head Conv2d regular weight groups use `weight_decay=0.0001`.
- Bias groups still receive bias warmup based on `strategy_role="bias"`.
- BatchNorm weights still avoid weight decay unless explicitly overridden and
  supported.

## Unsupported Strategy Combinations

While a training strategy is active, finetuning should be allowed to override
compatible optimizer parameters by default:

- `lr`
- `weight_decay`
- `momentum`
- `nesterov`
- scheduler scalar params that the strategy explicitly supports

Scheduler overrides under a strategy should be supported when the strategy can
apply them without changing its stepping semantics. If a scheduler override
would require independent optimizer stepping or a different scheduler class,
the strategy should reject it unless it explicitly supports that mode.

Finetuning should not silently switch optimizer class under a strategy.

Example unsupported config:

```yaml
trainer:
  training_strategy:
    name: TripleLRSGDStrategy

model:
  nodes:
    - name: Head
      finetuning:
        optimizer:
          name: AdamW
```

Recommended behavior:

- Raise a validation error.
- Message should say that `TripleLRSGDStrategy` only supports SGD-compatible
  finetuning overrides unless the strategy enables multi-optimizer mode.

Future extension:

- A strategy may opt into multi-optimizer mode.
- In that case, it owns how optimizers are stepped and how schedulers are
  monitored.

## Scheduler Semantics

Schedulers should be built and stepped according to the final optimization
plan.

Rules:

- One optimizer should have one scheduler by default.
- Multiple parameter groups inside one optimizer share that optimizer's
  scheduler.
- Multiple schedulers for one optimizer should require explicit support.
- Multiple optimizer classes are allowed without a training strategy.
- Optimizer instances should be minimized by merging compatible groups.
- `ReduceLROnPlateau` should use the exact metric key that validation logs.
- Scheduler configs must be grouped by full config identity, not just scheduler
  name.

Important fix:

When monitoring a main metric, use the formatted node name:

```text
val/metric/{formatted_node_name}/{metric_name}
```

not:

```text
val/metric/{raw_node_name}/{metric_name}
```

## Manual Optimization

Manual optimization should not be enabled just because more than one optimizer
exists unless the planner or strategy can preserve scheduler and accumulation
semantics.

Rough rule:

- Default path should prefer Lightning automatic optimization.
- If a plan requires manual optimization, that requirement must come from the
  strategy or from an explicitly supported multi-optimizer mode.
- `GradientAccumulationScheduler` should be rejected or handled explicitly when
  manual optimization is active.
- `ReduceLROnPlateau` should step from validation metrics, not stale training
  step state.

## Freezing and Unfreezing

Freezing should be part of the optimization plan, even if lifecycle actions are
performed by `TrainingManager`.

The planner should record:

- Which parameters start frozen.
- Which epoch unfreezes each parameter.
- Which optimizer receives the newly unfrozen parameters.
- Which LR should be used after unfreeze.

Initial recommendation:

- Keep parameters in groups from the start when possible.
- Toggle `requires_grad` on unfreeze.
- Avoid adding new param groups at runtime unless Lightning and all strategies
  handle the scheduler implications cleanly.
- This is the simpler path for a central planner because optimizer and
  scheduler topology is fixed before training starts.

## Logging and Debuggability

The planner should log a concise optimization summary:

- Number of optimizers.
- Optimizer type per optimizer.
- Scheduler type and monitor per optimizer.
- Parameter count per group.
- Group source.
- Group role tags.
- Overridden optimizer params.

This is important because finetuning configs are easy to mis-specify and hard
to inspect from optimizer objects alone.

## Migration Plan

1. Introduce `OptimizationPlanner` and data models without changing config
   schema.
2. Move current node-level optimizer extraction into the planner.
3. Make `configure_optimizers()` the only materialization point.
4. Make `configure_callbacks()` count optimizers from planner metadata, not by
   constructing optimizers.
5. Convert `TripleLRSGDStrategy` from optimizer factory to group transformer.
6. Move freezing metadata into the plan and update `TrainingManager`.
7. Add validation errors for unsupported strategy plus finetuning combinations.
8. Remove obsolete helper paths after tests cover the planner.

## Test Plan

Unit tests:

- Base config creates one optimizer and one scheduler.
- Node finetuning creates expected group params.
- Finetuning selector by module type works.
- Finetuning selector by parameter name works.
- Scheduler groups are not merged when scheduler params differ.
- `ReduceLROnPlateau` monitor key matches logged metric key.
- A parameter cannot appear in two optimizers by default.
- Strategy plus unsupported optimizer override raises a clear error.
- `TripleLRSGDStrategy` preserves finetuning LR and weight decay overrides.
- `TripleLRSGDStrategy` warmup uses role tags, not group indexes.

Integration tests:

- Existing default training still works.
- Existing `TripleLRSGDStrategy` tests still pass.
- OCR linear-layer finetuning replaces old `fc_decay` behavior.
- Frozen node unfreezes at the configured epoch with expected LR.
- Multiple node finetuning blocks train without duplicate parameter ownership.

## Decisions

1. Finetuning should allow multiple optimizer classes when no training strategy
   is active.
2. Finetuning under a training strategy may support scheduler overrides if the
   strategy can apply them without significant complexity or semantic changes.
3. Selector precedence is based on specificity. More specific selectors win.
   Config order breaks ties.
4. Frozen parameters should be present in optimizer groups from the start when
   possible, with `requires_grad` toggled at unfreeze time. This keeps the
   optimizer and scheduler topology static.
5. The public config key remains `finetuning`.
6. The planner should keep optimizer instance count as low as possible by
   merging parameters and groups with compatible optimizer and scheduler
   arguments.

## Rough Recommendation

Implement the planner first and keep the public config conservative.

The most stable first version is:

- Multiple optimizer classes are allowed when no training strategy is active,
  but compatible groups should be merged to keep optimizer count low.
- Finetuning may override optimizer params.
- Finetuning may override scheduler params under a strategy only when the
  strategy explicitly supports those overrides.
- Finetuning may not switch optimizer class under a training strategy.
- Strategy controls scheduler behavior when active.
- `TrainingManager` consumes planner metadata for freeze, unfreeze, and
  strategy lifecycle hooks.

That gives users a practical way to define further finetuning on top of
training strategies without making optimizer ownership ambiguous.
