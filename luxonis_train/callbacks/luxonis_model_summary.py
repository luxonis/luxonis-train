"""Prints the layer summary of the model, as a rich table or as plain
text.
"""

from io import StringIO
from typing import Any

from lightning.pytorch.callbacks import RichModelSummary
from lightning.pytorch.utilities.model_summary import get_human_readable_count
from loguru import logger
from rich.console import Console
from tabulate import tabulate
from typing_extensions import override


class LuxonisModelSummary(RichModelSummary):
    """Callback that prints the layer summary of the model.

    The callback extends the Lightning ``RichModelSummary`` and prints
    the summary when a fit starts. With ``rich`` on, it prints ``rich``
    tables to the console and writes a copy without terminal styling to
    the log file. With ``rich`` off, it logs a plain ``tabulate`` table
    and the totals. These records reach the console and the log file.
    `LuxonisLightningModule.configure_callbacks` adds this callback with
    ``max_depth=2``, and with ``rich`` equal to ``rich_logging`` of the
    config.

    """

    def __init__(self, rich: bool = True, **kwargs):
        """Initialize the callback.

        Args:
            rich (bool): Print ``rich`` tables. ``False`` logs a plain
                text table instead.
            **kwargs (``Any``): Keyword arguments for the Lightning
                ``RichModelSummary``. ``max_depth`` sets the deepest
                level of nested modules in the table, and ``0`` turns
                the summary off. Lightning passes every other keyword
                to `LuxonisModelSummary.summarize`.

        """
        super().__init__(**kwargs)

        self._rich = rich
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
        """Print the layer summary as ``rich`` tables or as plain text.

        The Lightning ``ModelSummary.on_fit_start`` hook calls this
        method at the start of a fit, on global rank 0 only. It skips
        the call when ``max_depth`` is ``0``.

        With ``rich`` on, the method prints a ``rich`` table and a grid
        of totals to the global ``rich`` console. It also writes both
        without terminal styling to the log file only. With ``rich``
        off, the method logs a ``tabulate`` table in the ``fancy_grid``
        format and one line per total with ``logger.info``. These lines
        reach the console and the log file.

        Each column of ``summary_data`` becomes a table column, in
        order. The method sets the headers by position: an index
        column, ``Name``, ``Type``, ``Params``, and ``Mode``. When
        ``summary_data`` has the columns ``In sizes`` and ``Out sizes``,
        the ``rich`` table adds these two headers after ``Mode``.
        Lightning puts a ``FLOPs`` column after ``Mode``. Thus these two
        headers go over the ``FLOPs`` and ``In sizes`` data. In the
        ``rich`` table, a data column after the last header gets no
        header. ``tabulate`` puts the five headers over the last five
        columns. With more than five columns, no header of the plain
        table is over its data.

        The totals are:

        - the trainable, the non-trainable, and the total parameter
          counts;
        - the estimated size of the parameters in MB, without the
          decimal part;
        - the number of modules in train mode and in eval mode.

        The counts and the size use a short form, for example
        ``1.2 M``.

        Args:
            *args (``Any``): The positional arguments of the Lightning
                hook, in order: ``summary_data``, the columns as
                ``(header, values)`` pairs; ``total_parameters``;
                ``trainable_parameters``; ``model_size``, in MB; and
                ``total_training_modes``, a dictionary with the keys
                ``"train"`` and ``"eval"``.
            **kwargs (``Any``): The keyword arguments of the Lightning
                hook. The ``rich`` table reads ``header_style``, which
                is ``"bold magenta"`` when it is not given. The method
                ignores all other keywords, such as ``total_flops``.

        """
        if self._rich:
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

        trainable, non_trainable, total, size = _format_parameter_counts(
            trainable_parameters, total_parameters, model_size
        )

        grid = Table.grid(expand=True)
        grid.add_column()
        grid.add_column()

        grid.add_row(f"[bold]Trainable params[/]: {trainable}")
        grid.add_row(f"[bold]Non-trainable params[/]: {non_trainable}")
        grid.add_row(f"[bold]Total params[/]: {total}")
        grid.add_row(
            f"[bold]Total estimated model params size (MB)[/]: {size}"
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

        trainable, non_trainable, total, size = _format_parameter_counts(
            trainable_parameters, total_parameters, model_size
        )
        logger.info(f"Trainable params: {trainable}")
        logger.info(f"Non-trainable params: {non_trainable}")
        logger.info(f"Total params: {total}")
        logger.info(f"Total estimated model params size (MB): {size}")
        logger.info(f"Modules in train mode: {total_training_modes['train']}")
        logger.info(f"Modules in eval mode: {total_training_modes['eval']}")


def _format_parameter_counts(
    trainable_parameters: int,
    total_parameters: int,
    model_size: float,
) -> list[str]:
    return [
        f"{get_human_readable_count(int(count)):<10}"
        for count in (
            trainable_parameters,
            total_parameters - trainable_parameters,
            total_parameters,
            model_size,
        )
    ]
