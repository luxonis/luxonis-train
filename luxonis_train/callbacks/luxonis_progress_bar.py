import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from io import StringIO
from typing import Any, TypedDict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ProgressBar,
    RichProgressBar,
    TQDMProgressBar,
)
from lightning.pytorch.utilities.types import (
    LRSchedulerConfig,
    LRSchedulerTypeUnion,
)
from loguru import logger
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tabulate import tabulate
from torch import Tensor, nn
from torch.optim import Optimizer
from typing_extensions import override

import luxonis_train as lxt
from luxonis_train.registry import CALLBACKS


class BaseLuxonisProgressBar(ABC, ProgressBar):
    _epoch_start_time: float

    @override
    def get_metrics(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> dict[str, int | str | float | dict[str, float]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        if "loss" in pl_module._loss_accumulators["train"]:
            items["Loss"] = pl_module._loss_accumulators["train"]["loss"]
        return items

    @abstractmethod
    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
        matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        """Print the results to the console.

        This includes the stage name, loss value, and tables with
        metrics.

        @type stage: str
        @param stage: Stage name.
        @type loss: float
        @param loss: Loss value.
        @type metrics: Mapping[str, Mapping[str, int | str | float]]
        @param metrics: Metrics in format {table_name: table}.
        @type matrices: Mapping[str, Mapping[str, Mapping[str, Any]]]
        @param matrices: Matrices in format {table_name: {name:
            matrix}}.
        """
        ...

    @abstractmethod
    def print_table(
        self,
        title: str,
        table: Iterable[tuple[str | int | float, ...]],
        column_names: list[str],
    ) -> None:
        """Print a table to the console.

        @type title: str
        @param title: Title of the table
        @type table: Iterable[tuple[str | int | float, ...]]
        @param table: Table to print as an iterable of rows, where each
            row is a tuple of values.
        @type column_names: list[str]
        @param column_names: Names of the columns in the table
        """
        ...

    def _log_progress(self, trainer: pl.Trainer) -> None:
        duration = (
            time.time() - self._epoch_start_time
            if hasattr(self, "_epoch_start_time")
            else 0.0
        )
        # Get last loss
        metrics = trainer.callback_metrics
        loss = metrics.get("train/loss")
        loss_str = f"{loss:.4f}" if loss else "N/A"

        # Log only to file
        logger.bind(file_only=True).info(
            f"[Epoch {trainer.current_epoch}/{trainer.max_epochs}] Duration: {duration:.2f}s | Train Loss: {loss_str}"
        )

    def format_matrix_for_printing(
        self, node: Any, name: str, value: Tensor
    ) -> dict[str, Any]:
        matrix = value.detach().cpu()
        rows, cols = matrix.shape

        row_labels = [str(i) for i in range(rows)]
        col_labels = [str(i) for i in range(cols)]

        module = getattr(node, "module", node)
        try:
            class_names = module.class_names
        except RuntimeError:
            class_names = []

        if len(class_names) == rows:
            row_labels = class_names
        elif len(class_names) + 1 == rows:
            row_labels = [*class_names, "no match"]

        if len(class_names) == cols:
            col_labels = class_names
        elif len(class_names) + 1 == cols:
            col_labels = [*class_names, "no match"]

        return {
            "values": matrix.tolist(),
            "row_labels": row_labels,
            "col_labels": col_labels,
            "row_axis": "GT",
            "col_axis": "Pred",
        }


@CALLBACKS.register()
class LuxonisTQDMProgressBar(TQDMProgressBar, BaseLuxonisProgressBar):
    """Custom text progress bar based on TQDMProgressBar from Pytorch
    Lightning.
    """

    def __init__(self):
        super().__init__(leave=True)

    @override
    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
        matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        self._rule(stage)
        logger.info(f"Loss: {loss}")
        logger.info("Metrics:")
        for table_name, table in metrics.items():
            self.print_table(
                table_name, list(table.items()), ["Name", "Value"]
            )
            for matrix_name, matrix in matrices.get(table_name, {}).items():
                self._print_matrix(
                    self._format_matrix_title(matrix_name), matrix
                )
        for table_name, table in matrices.items():
            if table_name in metrics:
                continue
            for matrix_name, matrix in table.items():
                self._print_matrix(
                    f"{table_name}/{self._format_matrix_title(matrix_name)}",
                    matrix,
                )
        self._rule()

    def _rule(self, title: str | None = None) -> None:
        if title is not None:
            logger.info(f"------{title}-----")
        else:
            logger.info("-----------------")

    @override
    def print_table(
        self,
        title: str,
        table: Iterable[tuple[str | int | float, ...]],
        column_names: list[str],
    ) -> None:
        """Print a table to the console using tabulate.

        @type title: str
        @param title: Title of the table
        @type table: Iterable[tuple[str | int | float, ...]]
        @param table: Table to print as an iterable of rows, where each
            row is a tuple of values.
        @type column_names: list[str]
        @param column_names: Names of the columns in the table
        """
        self._rule(title)
        formatted = tabulate(
            table,
            headers=column_names,
            tablefmt="fancy_grid",
            numalign="right",
        )
        logger.info(f"\n{formatted}\n")

    def _print_matrix(self, title: str, matrix: Mapping[str, Any]) -> None:
        values = matrix["values"]
        row_axis = matrix.get("row_axis", "Rows")
        col_axis = matrix.get("col_axis", "Cols")
        row_labels = matrix.get("row_labels") or [
            str(i) for i in range(len(values))
        ]
        col_labels = matrix.get("col_labels") or [
            str(i) for i in range(len(values[0]) if values else 0)
        ]
        rows = [[row_labels[i], *values[i]] for i in range(len(values))]
        self._rule(title)
        formatted = tabulate(
            rows,
            headers=[f"{row_axis} \\ {col_axis}", *list(col_labels)],
            tablefmt="fancy_grid",
            numalign="right",
        )
        logger.info(f"\n{formatted}\n")

    def _format_matrix_title(self, name: str) -> str:
        return name.replace("_", " ").title()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self._epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        super()._log_progress(trainer)


@CALLBACKS.register()
class LuxonisRichProgressBar(RichProgressBar, BaseLuxonisProgressBar):
    """Custom rich text progress bar based on RichProgressBar from
    Pytorch Lightning.
    """

    def __init__(self):
        super().__init__(leave=True)
        self._log_buffer = StringIO()
        self._log_console = Console(
            file=self._log_buffer, force_terminal=False
        )

    @property
    def console(self) -> Console:
        if self._console is None:  # pragma: no cover
            raise RuntimeError(
                "Console is not initialized for the `LuxonisRichProgressBar`. "
                "Consider setting `rich_logging` to `False` in the configuration."
            )
        return self._console

    @override
    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
        matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        # Terminal output
        self.console.rule(f"{stage}", style="bold magenta")
        self.console.print(
            f"[bold magenta]Loss:[/bold magenta] [white]{loss}[/white]"
        )
        self.console.print("[bold magenta]Metrics:[/bold magenta]")
        self._print_result_tables(metrics, matrices)
        self.console.rule(style="bold magenta")

        # Log file output
        self._log_console.rule(f"{stage}")
        self._log_console.print(f"Loss: {loss}")
        self._log_console.print("Metrics:")
        self._print_result_tables(metrics, matrices, self._log_console)
        self._log_console.rule()

        # Dump to logger
        logger.bind(file_only=True).info("\n" + self._log_buffer.getvalue())
        self._log_buffer.seek(0)
        self._log_buffer.truncate(0)

    def _print_result_tables(
        self,
        metrics: Mapping[str, Mapping[str, int | str | float]],
        matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
        console: Console | None = None,
    ) -> None:
        for table_name, table in metrics.items():
            self.print_table(
                table_name,
                list(table.items()),
                ["Name", "Value"],
                console=console,
            )
            for matrix_name, matrix in matrices.get(table_name, {}).items():
                self._print_matrix(
                    self._format_matrix_title(matrix_name),
                    matrix,
                    console=console,
                )
        for table_name, table in matrices.items():
            if table_name in metrics:
                continue
            for matrix_name, matrix in table.items():
                self._print_matrix(
                    f"{table_name}/{self._format_matrix_title(matrix_name)}",
                    matrix,
                    console=console,
                )

    @override
    def print_table(
        self,
        title: str,
        table: Iterable[tuple[str | int | float, ...]],
        column_names: list[str],
        console: Console | None = None,
    ) -> None:
        """Print a table to the console using rich text.

        @type title: str
        @param title: Title of the table
        @type table: Iterable[tuple[str | int | float, ...]]
        @param table: Table to print as an iterable of rows, where each
            row is a tuple of values.
        @type column_names: list[str]
        @param column_names: Names of the columns in the table
        @param console: Console instance to use, if None use default
            console. Defaults to None.
        @type console: Console | None
        """
        console = console or self.console
        rich_table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            title_style="bold",
        )
        for i, column_name in enumerate(column_names):
            rich_table.add_column(
                column_name, style="magenta" if i == 0 else "white"
            )
        for name, *values in table:
            rich_table.add_row(
                str(name),
                *[
                    f"{value:.5f}" if isinstance(value, float) else str(value)
                    for value in values
                ],
            )
        console.print(rich_table)

    def _print_matrix(
        self,
        title: str,
        matrix: Mapping[str, Any],
        console: Console | None = None,
    ) -> None:
        console = console or self.console
        values = matrix["values"]
        row_axis = matrix.get("row_axis", "Rows")
        col_axis = matrix.get("col_axis", "Cols")
        row_labels = matrix.get("row_labels") or [
            str(i) for i in range(len(values))
        ]
        col_labels = matrix.get("col_labels") or [
            str(i) for i in range(len(values[0]) if values else 0)
        ]

        rich_table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            title_style="italic",
        )
        rich_table.add_column(f"{row_axis} \\ {col_axis}", style="magenta")
        for col in col_labels:
            rich_table.add_column(str(col), style="white", justify="right")
        for idx, row in enumerate(values):
            label = row_labels[idx] if idx < len(row_labels) else str(idx)
            rich_table.add_row(label, *[str(v) for v in row])
        console.print(rich_table)

    def _format_matrix_title(self, name: str) -> str:
        return name.replace("_", " ").title()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        super().on_train_epoch_start(trainer, pl_module)
        self._epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        super()._log_progress(trainer)


def build_optimizer_summary(
    optimizers: Sequence[Optimizer],
    schedulers: Sequence[LRSchedulerTypeUnion | LRSchedulerConfig],
    modules: Mapping[str, nn.Module],
) -> dict[str, Any]:
    """Build the serializable optimizer / parameter-group summary.

    Two different denominators are used, chosen so that percentages sum
    naturally in the axis the reader cares about:

        - B{Group-level} percentages are relative to all model parameters,
          so summing across all groups of all optimizers gives 100% (modulo
          unclaimed / external parameters).
        - B{Owner-level} percentages inside each group are relative to all
          parameters belonging to that owner, so summing all appearances of
          a single owner across the optimizers gives 100% — telling the
          reader how each node's parameters were split across groups.

    Frozen parameters remain assigned to optimizer groups by the static
    training plan. Assignment percentages therefore include them, while
    every group separately reports how much of its assignment is currently
    trainable or frozen.
    """
    stats = _collect_owner_stats(modules, optimizers)
    return {
        "n_optimizers": len(optimizers),
        "model_tensors": stats.model_tensors,
        "model_params": stats.model_params,
        "trainable_tensors": stats.trainable_tensors,
        "trainable_params": stats.trainable_params,
        "frozen_tensors": stats.frozen_tensors,
        "frozen_params": stats.frozen_params,
        "optimizers": [
            _summarize_optimizer(i, optimizer, scheduler, stats)
            for i, (optimizer, scheduler) in enumerate(
                zip(optimizers, schedulers, strict=True)
            )
        ],
    }


def log_optimizer_summary(
    summary: dict[str, Any], use_rich: bool = True
) -> None:
    """Render the optimizer / parameter-group summary.

    Emits a pretty console version (nested rich panels, or a plaintext
    indented-list fallback) and dumps an equivalent JSON payload to the
    log file via ``logger.bind(file_only=True)``.
    """
    if use_rich:
        _render_optimizer_summary_rich(summary)
    else:
        _render_optimizer_summary_plain(summary)

    logger.bind(file_only=True).info(
        "Optimizer / parameter-group summary (JSON):\n"
        + json.dumps(summary, indent=2, default=str)
    )


class _OwnerInfo(TypedDict):
    name: str
    n_tensors: int
    n_tensors_of_owner: int
    tensors_pct_of_owner: float
    n_params: int
    n_params_of_owner: int
    params_pct_of_owner: float
    trainable_tensors: int
    trainable_params: int
    frozen_tensors: int
    frozen_params: int


class _GroupInfo(TypedDict):
    index: int
    n_tensors: int
    n_params: int
    trainable_tensors: int
    trainable_params: int
    frozen_tensors: int
    frozen_params: int
    tensors_pct_of_model: float
    params_pct_of_model: float
    hyperparams: dict[str, object]
    owners: list[_OwnerInfo]


class _OptimizerInfo(TypedDict):
    index: int
    optimizer: str
    scheduler: str
    n_groups: int
    groups: list[_GroupInfo]


class _OwnerStats:
    """Per-owner parameter tallies, deduplicated by parameter
    identity.
    """

    def __init__(self):
        self.param_owner: dict[int, str] = {}
        self.owner_tensors: dict[str, int] = defaultdict(int)
        self.owner_params: dict[str, int] = defaultdict(int)
        self.owner_trainable_tensors: dict[str, int] = defaultdict(int)
        self.owner_trainable_params: dict[str, int] = defaultdict(int)
        self.frozen_tensors = 0
        self.frozen_params = 0

    @property
    def model_tensors(self) -> int:
        return sum(self.owner_tensors.values())

    @property
    def model_params(self) -> int:
        return sum(self.owner_params.values())

    @property
    def trainable_tensors(self) -> int:
        return sum(self.owner_trainable_tensors.values())

    @property
    def trainable_params(self) -> int:
        return sum(self.owner_trainable_params.values())

    def add(self, owner: str, param: Tensor) -> None:
        if id(param) in self.param_owner:
            return
        self.param_owner[id(param)] = owner
        self.owner_tensors[owner] += 1
        self.owner_params[owner] += param.numel()
        if param.requires_grad:
            self.owner_trainable_tensors[owner] += 1
            self.owner_trainable_params[owner] += param.numel()
        else:
            self.frozen_tensors += 1
            self.frozen_params += param.numel()


def _collect_owner_stats(
    modules: Mapping[str, nn.Module], optimizers: Sequence[Optimizer]
) -> _OwnerStats:
    stats = _OwnerStats()
    for owner_name, module in modules.items():
        for param in module.parameters():
            stats.add(owner_name, param)
    # A parameter can sit in an optimizer group without a known module.
    # Count it under "<external>" so it gets real denominators and
    # trainability totals.
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for param in group["params"]:
                stats.add("<external>", param)
    return stats


def _summarize_optimizer(
    index: int,
    optimizer: Optimizer,
    scheduler: LRSchedulerTypeUnion | LRSchedulerConfig,
    stats: _OwnerStats,
) -> _OptimizerInfo:
    if isinstance(scheduler, dict):
        scheduler_name = type(scheduler["scheduler"]).__name__
    else:
        scheduler_name = type(scheduler).__name__
    return {
        "index": index,
        "optimizer": type(optimizer).__name__,
        "scheduler": scheduler_name,
        "n_groups": len(optimizer.param_groups),
        "groups": [
            _summarize_group(
                g_idx, group["params"], _group_hyperparams(group), stats
            )
            for g_idx, group in enumerate(optimizer.param_groups)
        ],
    }


def _summarize_group(
    index: int,
    params: Sequence[Tensor],
    hyperparams: dict[str, object],
    stats: _OwnerStats,
) -> _GroupInfo:
    per_owner_tensors: dict[str, int] = defaultdict(int)
    per_owner_numel: dict[str, int] = defaultdict(int)
    per_owner_trainable_tensors: dict[str, int] = defaultdict(int)
    per_owner_trainable_numel: dict[str, int] = defaultdict(int)
    total_numel = 0
    total_trainable_tensors = 0
    total_trainable_numel = 0
    for p in params:
        owner = stats.param_owner.get(id(p), "<external>")
        per_owner_tensors[owner] += 1
        per_owner_numel[owner] += p.numel()
        total_numel += p.numel()
        if p.requires_grad:
            per_owner_trainable_tensors[owner] += 1
            per_owner_trainable_numel[owner] += p.numel()
            total_trainable_tensors += 1
            total_trainable_numel += p.numel()

    owners: list[_OwnerInfo] = [
        {
            "name": name,
            "n_tensors": per_owner_tensors[name],
            "n_tensors_of_owner": stats.owner_tensors[name],
            "tensors_pct_of_owner": _pct(
                per_owner_tensors[name], stats.owner_tensors[name]
            ),
            "n_params": per_owner_numel[name],
            "n_params_of_owner": stats.owner_params[name],
            "params_pct_of_owner": _pct(
                per_owner_numel[name], stats.owner_params[name]
            ),
            "trainable_tensors": per_owner_trainable_tensors[name],
            "trainable_params": per_owner_trainable_numel[name],
            "frozen_tensors": per_owner_tensors[name]
            - per_owner_trainable_tensors[name],
            "frozen_params": per_owner_numel[name]
            - per_owner_trainable_numel[name],
        }
        for name in sorted(
            per_owner_numel, key=lambda n: per_owner_numel[n], reverse=True
        )
    ]
    n_tensors = len(params)
    return {
        "index": index,
        "n_tensors": n_tensors,
        "n_params": total_numel,
        "trainable_tensors": total_trainable_tensors,
        "trainable_params": total_trainable_numel,
        "frozen_tensors": n_tensors - total_trainable_tensors,
        "frozen_params": total_numel - total_trainable_numel,
        "tensors_pct_of_model": _pct(n_tensors, stats.model_tensors),
        "params_pct_of_model": _pct(total_numel, stats.model_params),
        "hyperparams": hyperparams,
        "owners": owners,
    }


def _group_hyperparams(group: Mapping[str, object]) -> dict[str, object]:
    return {
        k: v
        for k, v in group.items()
        if k != "params"
        and not callable(v)
        and not isinstance(v, (list, tuple, dict))
    }


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def _render_optimizer_summary_rich(summary: dict[str, Any]) -> None:
    from rich import get_console
    from rich.console import Group

    console = get_console()
    console.print(
        Panel.fit(
            f"[bold]Using {summary['n_optimizers']} optimizer(s)[/]  "
            f"[dim]trainable: {summary['trainable_tensors']:,} tensors / "
            f"{summary['trainable_params']:,} params  |  "
            f"frozen: {summary['frozen_tensors']:,} tensors / "
            f"{summary['frozen_params']:,} params[/]",
            border_style="cyan",
        )
    )
    for opt in summary["optimizers"]:
        group_panels: list[RenderableType] = []
        for group in opt["groups"]:
            header_line = (
                f"[white]{group['n_tensors']} tensors[/] "
                f"[dim]({group['tensors_pct_of_model']:.1f}% of model)[/]"
                f"  •  "
                f"[white]{group['n_params']:,} params[/] "
                f"[dim]({group['params_pct_of_model']:.1f}% of model)[/]"
            )
            activity_line = (
                f"[green]trainable: {group['trainable_tensors']:,} tensors / "
                f"{group['trainable_params']:,} params[/]  •  "
                f"[dim]frozen: {group['frozen_tensors']:,} tensors / "
                f"{group['frozen_params']:,} params[/]"
            )
            # `Columns` always measures as wide as the console, which would
            # stop the enclosing `Panel.fit`s from shrinking to their
            # content. A grid measures its actual width.
            side_by_side = Table.grid(padding=(0, 1))
            side_by_side.add_column()
            side_by_side.add_column()
            side_by_side.add_row(
                _render_hyperparam_panel(group["hyperparams"]),
                _render_owners_panel(group["owners"]),
            )
            group_panels.append(
                Panel.fit(
                    Group(
                        Text(""),
                        header_line,
                        activity_line,
                        Text(""),
                        side_by_side,
                    ),
                    title=f"[bold]Group #{group['index']}[/]",
                    title_align="left",
                    border_style="blue",
                )
            )

        opt_body = Group(
            Text(""),
            Text.from_markup(
                f"[cyan]{opt['optimizer']}[/] + "
                f"[magenta]{opt['scheduler']}[/]  "
                f"[dim]({opt['n_groups']} parameter group(s))[/]"
            ),
            Text(""),
            *group_panels,
        )
        console.print(
            Panel.fit(
                opt_body,
                title=f"[bold]Optimizer #{opt['index']}[/]",
                title_align="left",
                border_style="magenta",
            )
        )


def _render_optimizer_summary_plain(summary: dict[str, Any]) -> None:
    lines = [
        f"Using {summary['n_optimizers']} optimizer(s).",
        (
            f"  trainable: {summary['trainable_tensors']:,} tensors / "
            f"{summary['trainable_params']:,} params"
        ),
        (
            f"  frozen:    {summary['frozen_tensors']:,} tensors / "
            f"{summary['frozen_params']:,} params"
        ),
    ]
    for opt in summary["optimizers"]:
        lines.append("")
        lines.append(
            f"Optimizer #{opt['index']}: {opt['optimizer']} + "
            f"{opt['scheduler']}  ({opt['n_groups']} parameter group(s))"
        )
        for group in opt["groups"]:
            lines.extend(_plain_group_lines(group))
    logger.info("\n" + "\n".join(lines) + "\n")


def _plain_group_lines(group: _GroupInfo) -> list[str]:
    lines = [
        (
            f"  Group #{group['index']}: "
            f"{group['n_tensors']} tensors "
            f"({group['tensors_pct_of_model']:.1f}% of model)  •  "
            f"{group['n_params']:,} params "
            f"({group['params_pct_of_model']:.1f}% of model)"
        ),
        (
            f"    trainable: {group['trainable_tensors']:,} tensors / "
            f"{group['trainable_params']:,} params  |  "
            f"frozen: {group['frozen_tensors']:,} tensors / "
            f"{group['frozen_params']:,} params"
        ),
        "    hyperparameters:",
    ]
    if group["hyperparams"]:
        lines.extend(
            f"      {k} = {_format_hyperparam(v)}"
            for k, v in group["hyperparams"].items()
        )
    else:
        lines.append("      -")
    lines.append("    owners:")
    if group["owners"]:
        for owner in group["owners"]:
            lines.extend(_plain_owner_lines(owner))
    else:
        lines.append("      -")
    return lines


def _plain_owner_lines(o: _OwnerInfo) -> list[str]:
    return [
        f"      {o['name']}",
        (
            f"        tensors "
            f"{o['n_tensors']}/{o['n_tensors_of_owner']} "
            f"({o['tensors_pct_of_owner']:.1f}% of owner)"
        ),
        (
            f"        params  "
            f"{o['n_params']:,}/{o['n_params_of_owner']:,} "
            f"({o['params_pct_of_owner']:.1f}% of owner)"
        ),
        (
            f"        trainable {o['trainable_tensors']:,} "
            f"tensors / {o['trainable_params']:,} params  |  "
            f"frozen {o['frozen_tensors']:,} tensors / "
            f"{o['frozen_params']:,} params"
        ),
    ]


def _format_hyperparam(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _render_hyperparam_panel(hyperparams: dict[str, Any]) -> RenderableType:
    if not hyperparams:
        return Panel("[dim]-[/]", border_style="dim", padding=(0, 1))

    grid = Table.grid(padding=(0, 1))
    grid.add_column(justify="right", style="bold yellow", no_wrap=True)
    grid.add_column(style="dim")
    grid.add_column(style="white")
    for k, v in hyperparams.items():
        grid.add_row(str(k), "=", _format_hyperparam(v))
    return Panel(
        grid,
        title="[bold yellow]hyperparameters[/]",
        title_align="left",
        border_style="yellow",
        padding=(0, 1),
    )


def _render_owners_panel(owners: list[dict[str, Any]]) -> RenderableType:
    if not owners:
        return Panel("[dim]-[/]", border_style="dim", padding=(0, 1))

    outer = Table.grid(padding=(0, 0))
    outer.add_column()
    for i, o in enumerate(owners):
        stat_grid = Table.grid(padding=(0, 1))
        stat_grid.add_column(justify="right", style="bold cyan", no_wrap=True)
        stat_grid.add_column(justify="right", style="white")
        stat_grid.add_column(style="dim")
        stat_grid.add_row(
            "tensors",
            f"{o['n_tensors']}/{o['n_tensors_of_owner']}",
            f"({o['tensors_pct_of_owner']:.1f}%)",
        )
        stat_grid.add_row(
            "params",
            f"{o['n_params']:,}/{o['n_params_of_owner']:,}",
            f"({o['params_pct_of_owner']:.1f}%)",
        )
        stat_grid.add_row(
            "trainable",
            f"{o['trainable_tensors']:,} tensors",
            f"{o['trainable_params']:,} params",
        )
        stat_grid.add_row(
            "frozen",
            f"{o['frozen_tensors']:,} tensors",
            f"{o['frozen_params']:,} params",
        )
        outer.add_row(Text(o["name"], style="bold green"))
        outer.add_row(stat_grid)
        if i < len(owners) - 1:
            outer.add_row(Text(""))
    return Panel(
        outer,
        title="[bold green]owners[/]",
        title_align="left",
        border_style="green",
        padding=(0, 1),
    )
