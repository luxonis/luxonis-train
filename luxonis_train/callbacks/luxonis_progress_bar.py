from collections.abc import Mapping

import lightning.pytorch as pl
from lightning.pytorch.callbacks import RichProgressBar
from rich.console import Console
from rich.table import Table

from luxonis_train.utils.registry import CALLBACKS


@CALLBACKS.register_module()
class LuxonisProgressBar(RichProgressBar):
    """Custom rich text progress bar based on RichProgressBar from Pytorch Lightning."""

    _console: Console

    def __init__(self):
        super().__init__(leave=True)

    def print_single_line(self, text: str, style: str = "magenta") -> None:
        """Prints single line of text to the console.

        @type text: str
        @param text: Text to print.
        @type style: str
        @param style: Style of the text. Defaults to C{"magenta"}.
        """
        if self._check_console():
            self._console.print(f"[{style}]{text}[/{style}]")
        else:
            print(text)

    def get_metrics(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> dict[str, int | str | float | dict[str, float]]:
        # NOTE: there might be a cleaner way of doing this
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        if trainer.training and pl_module.training_step_outputs:
            items["Loss"] = pl_module.training_step_outputs[-1]["loss"].item()
        return items

    def _check_console(self) -> bool:
        """Checks if console is set.

        @rtype: bool
        @return: True if console is set, False otherwise.
        """
        return self._console is not None

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
        @param value_name: Name of the value column. Defaults to C{"Value"}.
        """
        if self._check_console():
            rich_table = Table(
                title=title,
                show_header=True,
                header_style="bold magenta",
            )
            rich_table.add_column(key_name, style="magenta")
            rich_table.add_column(value_name, style="white")
            for name, value in table.items():
                if isinstance(value, float):
                    rich_table.add_row(name, f"{value:.5f}")
                else:
                    rich_table.add_row(name, str(value))
            self._console.print(rich_table)
        else:
            print(f"------{title}-----")
            for name, value in table.items():
                print(f"{name}: {value}")

    def print_tables(
        self, tables: Mapping[str, Mapping[str, int | str | float]]
    ) -> None:
        """Prints multiple tables to the console using rich text.

        @type tables: Mapping[str, Mapping[str, int | str | float]]
        @param tables: Tables to print in format {table_name: table}.
        """
        for table_name, table in tables.items():
            self.print_table(table_name, table)

    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
    ) -> None:
        """Prints results to the console using rich text.

        @type stage: str
        @param stage: Stage name.
        @type loss: float
        @param loss: Loss value.
        @type metrics: Mapping[str, Mapping[str, int | str | float]]
        @param metrics: Metrics in format {table_name: table}.
        """
        if self._check_console():
            self._console.rule(f"{stage}", style="bold magenta")
            self._console.print(
                f"[bold magenta]Loss:[/bold magenta] [white]{loss}[/white]"
            )
            self._console.print("[bold magenta]Metrics:[/bold magenta]")
            self.print_tables(metrics)
            self._console.rule(style="bold magenta")
        else:
            print(f"------{stage}-----")
            print(f"Loss: {loss}")

            for node_name, node_metrics in metrics.items():
                for metric_name, metric_value in node_metrics.items():
                    print(
                        f"{stage} metric: {node_name}/{metric_name}: {metric_value:.4f}"
                    )
