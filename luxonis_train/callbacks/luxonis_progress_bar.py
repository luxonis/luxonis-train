import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from io import StringIO
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    ProgressBar,
    RichProgressBar,
    TQDMProgressBar,
)
from loguru import logger
from rich.console import Console
from rich.table import Table
from tabulate import tabulate
from torch import Tensor
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
        """Prints results to the console.

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
            row_labels = [*class_names, "background"]

        if len(class_names) == cols:
            col_labels = class_names
        elif len(class_names) + 1 == cols:
            col_labels = [*class_names, "background"]

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
    Lightning."""

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
            self._print_table(table_name, table)
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

    def _print_table(
        self,
        title: str,
        table: Mapping[str, int | str | float],
        key_name: str = "Name",
        value_name: str = "Value",
    ) -> None:
        """Prints table to the console using tabulate.

        @type title: str
        @param title: Title of the table
        @type table: Mapping[str, int | str | float]
        @param table: Table to print
        @type key_name: str
        @param key_name: Name of the key column. Defaults to C{"Name"}.
        @type value_name: str
        @param value_name: Name of the value column. Defaults to
            C{"Value"}.
        """
        self._rule(title)
        formatted = tabulate(
            table.items(),
            headers=[key_name, value_name],
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
    Pytorch Lightning."""

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
        for table_name, table in metrics.items():
            self._print_table(table_name, table)
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
        self.console.rule(style="bold magenta")

        # Log file output
        self._log_console.rule(f"{stage}")
        self._log_console.print(f"Loss: {loss}")
        self._log_console.print("Metrics:")
        for table_name, table in metrics.items():
            self._print_table(table_name, table, console=self._log_console)
            for matrix_name, matrix in matrices.get(table_name, {}).items():
                self._print_matrix(
                    self._format_matrix_title(matrix_name),
                    matrix,
                    console=self._log_console,
                )
        for table_name, table in matrices.items():
            if table_name in metrics:
                continue
            for matrix_name, matrix in table.items():
                self._print_matrix(
                    f"{table_name}/{self._format_matrix_title(matrix_name)}",
                    matrix,
                    console=self._log_console,
                )
        self._log_console.rule()

        # Dump to logger
        logger.bind(file_only=True).info("\n" + self._log_buffer.getvalue())
        self._log_buffer.seek(0)
        self._log_buffer.truncate(0)

    def _print_table(
        self,
        title: str,
        table: Mapping[str, int | str | float],
        key_name: str = "Name",
        value_name: str = "Value",
        console: Console | None = None,
    ) -> None:
        """Prints table to the console using rich text.

        @type title: str
        @param title: Title of the table
        @type table: Mapping[str, int | str | float]
        @param table: Table to print
        @type key_name: str
        @param key_name: Name of the key column. Defaults to C{"Name"}.
        @type value_name: str
        @param value_name: Name of the value column. Defaults to
            C{"Value"}.
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
        rich_table.add_column(key_name, style="magenta")
        rich_table.add_column(value_name, style="white")
        for name, value in table.items():
            rich_table.add_row(name, f"{value:.5f}")
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
