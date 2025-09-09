from io import StringIO
from typing import Any

from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.utilities.model_summary import get_human_readable_count
from loguru import logger
from rich.console import Console
from tabulate import tabulate
from typing_extensions import override


class LuxonisModelSummary(RichModelSummary):
    def __init__(self, rich: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.rich = rich
        self._log_buffer = StringIO()
        self._log_console = Console(
            file=self._log_buffer, force_terminal=False
        )

    @override
    def summarize(
        self,
        *args,
        **kwargs,
    ) -> None:
        if self.rich:
            self._rich_summarize(*args, **kwargs)
        else:
            self._regular_summarize(*args, **kwargs)

    def _rich_summarize(
        self,
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **summarize_kwargs: Any,
    ) -> None:
        from rich import get_console
        from rich.table import Table

        console = get_console()

        header_style: str = summarize_kwargs.get(
            "header_style", "bold magenta"
        )
        table = Table(header_style=header_style)
        table.add_column(" ", style="dim")
        table.add_column("Name", justify="left", no_wrap=True)
        table.add_column("Type")
        table.add_column("Params", justify="right")
        table.add_column("Mode")

        column_names = next(zip(*summary_data, strict=True))

        for column_name in ["In sizes", "Out sizes"]:
            if column_name in column_names:
                table.add_column(column_name, justify="right", style="white")

        rows = list(zip(*(arr[1] for arr in summary_data), strict=True))
        for row in rows:
            table.add_row(*row)

        console.print(table)
        self._log_console.print(table)

        parameters = []
        for param in [
            trainable_parameters,
            total_parameters - trainable_parameters,
            total_parameters,
            model_size,
        ]:
            parameters.append(
                "{:<{}}".format(get_human_readable_count(int(param)), 10)
            )

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {parameters[0]}")
        grid.add_row(f"[bold]Non-trainable params[/]: {parameters[1]}")
        grid.add_row(f"[bold]Total params[/]: {parameters[2]}")
        grid.add_row(
            f"[bold]Total estimated model params size (MB)[/]: {parameters[3]}"
        )
        grid.add_row(
            f"[bold]Modules in train mode[/]: {total_training_modes['train']}"
        )
        grid.add_row(
            f"[bold]Modules in eval mode[/]: {total_training_modes['eval']}"
        )

        console.print(grid)
        self._log_console.print(grid)

        logger.bind(file_only=True).info("\n" + self._log_buffer.getvalue())
        self._log_buffer.seek(0)
        self._log_buffer.truncate(0)

    def _regular_summarize(
        self,
        summary_data: list[tuple[str, list[str]]],
        total_parameters: int,
        trainable_parameters: int,
        model_size: float,
        total_training_modes: dict[str, int],
        **_,
    ) -> None:
        rows = list(zip(*(arr[1] for arr in summary_data), strict=True))
        table = tabulate(
            rows,
            headers=[" ", "Name", "Type", "Params", "Mode"],
            tablefmt="fancy_grid",
        )
        logger.info(f"\n{table}\n")

        parameters = []
        for param in [
            trainable_parameters,
            total_parameters - trainable_parameters,
            total_parameters,
            model_size,
        ]:
            parameters.append(
                "{:<{}}".format(get_human_readable_count(int(param)), 10)
            )
        logger.info(f"Trainable params: {parameters[0]}")
        logger.info(f"Non-trainable params: {parameters[1]}")
        logger.info(f"Total params: {parameters[2]}")
        logger.info(f"Total estimated model params size (MB): {parameters[3]}")
        logger.info(f"Modules in train mode: {total_training_modes['train']}")
        logger.info(f"Modules in eval mode: {total_training_modes['eval']}")
