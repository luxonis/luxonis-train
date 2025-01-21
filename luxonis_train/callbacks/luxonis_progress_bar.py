from abc import ABC, abstractmethod
from collections.abc import Mapping

import lightning.pytorch as pl
import tabulate
from lightning.pytorch.callbacks import (
    ProgressBar,
    RichProgressBar,
    TQDMProgressBar,
)
from rich.console import Console
from rich.table import Table

from luxonis_train.utils.registry import CALLBACKS


class BaseLuxonisProgressBar(ABC, ProgressBar):
    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        if trainer.training and pl_module.training_step_outputs:
            items["Loss"] = pl_module.training_step_outputs[-1]["loss"].item()
        return items

    @abstractmethod
    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
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
        """
        ...


@CALLBACKS.register()
class LuxonisTQDMProgressBar(TQDMProgressBar, BaseLuxonisProgressBar):
    """Custom text progress bar based on TQDMProgressBar from Pytorch
    Lightning."""

    def __init__(self):
        super().__init__(leave=True)

    def _rule(self, title: str | None = None) -> None:
        if title is not None:
            print(f"------{title}-----")
        else:
            print("-----------------")

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
        print(
            tabulate.tabulate(
                table.items(),
                headers=[key_name, value_name],
                tablefmt="fancy_grid",
                numalign="right",
            )
        )
        print()

    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
    ) -> None:
        self._rule(stage)
        print(f"Loss: {loss}")
        print("Metrics:")
        for table_name, table in metrics.items():
            self._print_table(table_name, table)
        self._rule()


@CALLBACKS.register()
class LuxonisRichProgressBar(RichProgressBar, BaseLuxonisProgressBar):
    """Custom rich text progress bar based on RichProgressBar from
    Pytorch Lightning."""

    def __init__(self):
        super().__init__(leave=True)

    @property
    def console(self) -> Console:
        if self._console is None:  # pragma: no cover
            raise RuntimeError(
                "Console is not initialized for the `LuxonisRichProgressBar`. "
                "Consider setting `tracker.use_rich_progress_bar` to `False` in the configuration."
            )
        return self._console

    def print_table(
        self,
        title: str,
        table: Mapping[str, int | str | float],
        key_name: str = "Name",
        value_name: str = "Value",
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
        """
        rich_table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
        )
        rich_table.add_column(key_name, style="magenta")
        rich_table.add_column(value_name, style="white")
        for name, value in table.items():
            rich_table.add_row(name, f"{value:.5f}")
        self.console.print(rich_table)

    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
    ) -> None:
        self.console.rule(f"{stage}", style="bold magenta")
        self.console.print(
            f"[bold magenta]Loss:[/bold magenta] [white]{loss}[/white]"
        )
        self.console.print("[bold magenta]Metrics:[/bold magenta]")
        for table_name, table in metrics.items():
            self.print_table(table_name, table)
        self.console.rule(style="bold magenta")
