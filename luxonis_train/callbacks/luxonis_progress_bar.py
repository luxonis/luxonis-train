"""The progress bars of the trainer, and the optimizer summary.

`LuxonisRichProgressBar` and `LuxonisTQDMProgressBar` show the batch
progress and print the results of each evaluation epoch. Both bars
mirror the printed results to the log file.

`build_optimizer_summary` and `log_optimizer_summary` report how the
parameters of the model are split across the optimizers and their
parameter groups. The summary shows the effect of the finetuning rules
and of the training strategy of the config.

"""

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
    """Base class for the progress bars of the trainer.

    The class drops the ``v_num`` item from the bar and adds the
    running mean of the train loss as ``Loss``. A subclass prints the
    results of an evaluation epoch with
    `BaseLuxonisProgressBar.print_results` and
    `BaseLuxonisProgressBar.print_table`. The subclasses also write one
    summary line per train epoch to the log file.

    """

    _epoch_start_time: float

    @override
    def get_metrics(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> dict[str, int | str | float | dict[str, float]]:
        """Return the items shown at the end of the progress bar.

        The items are the metrics that the model logs with
        ``prog_bar=True``, as ``ProgressBar.get_metrics`` collects
        them, without the ``v_num`` entry. When the train loss
        accumulator of ``pl_module`` holds a ``"loss"`` entry, the
        method adds it as ``Loss``. That value is the running mean of
        the total loss over the batches of the current epoch so far.

        Args:
            trainer (``pl.Trainer``): The trainer.
            pl_module (LuxonisLightningModule): The model. Its train
                loss accumulator provides the ``Loss`` value.

        Returns:
            dict[str, int | str | float | dict[str, float]]: The items
            to show, keyed by name.

        """
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
        """Print the results of an evaluation epoch.

        An implementation must print the stage name, the loss, one
        table per node in ``metrics``, and one table per matrix in
        ``matrices``.

        Args:
            stage (str): Name of the stage, for example ``"Validation"``.
            loss (float): Mean loss of the epoch.
            metrics (``Mapping[str, Mapping[str, int | str | float]]``):
                Scalar metrics as ``{node_name: {metric_name: value}}``.
            matrices (``Mapping[str, Mapping[str, Mapping[str, Any]]]``):
                Matrix metrics as ``{node_name: {metric_name: matrix}}``.
                Each matrix is a dictionary in the format of
                `BaseLuxonisProgressBar.format_matrix_for_printing`.

        """
        ...

    @abstractmethod
    def print_table(
        self,
        title: str,
        table: Iterable[tuple[str | int | float, ...]],
        column_names: list[str],
    ) -> None:
        """Print one table.

        An implementation must print ``title``, then a table with one
        header per entry of ``column_names`` and one row per tuple of
        ``table``.

        Args:
            title (str): Title of the table.
            table (``Iterable[tuple[str | int | float, ...]]``): The rows.
                Each row is a tuple with one value per column.
            column_names (list[str]): Names of the columns.

        """
        ...

    def _log_progress(self, trainer: pl.Trainer) -> None:
        duration = (
            time.time() - self._epoch_start_time
            if hasattr(self, "_epoch_start_time")
            else 0.0
        )
        # The module logs `train/loss` after this hook, so this is the
        # value of the previous epoch, or `None` in the first one.
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
        """Convert a matrix metric into a printable dictionary.

        The row and column labels are the class names of ``node`` when
        the matrix has one row (column) per class. When the matrix has
        one extra row (column), the labels are the class names plus
        ``"no match"``. Otherwise the labels are the indices as
        strings. When the class names of ``node`` raise a
        ``RuntimeError``, for example because the node has no dataset
        metadata, the class names count as empty. Any other exception
        of that lookup propagates.

        The result has the keys ``"values"`` (the matrix as a nested
        list), ``"row_labels"``, ``"col_labels"``, ``"row_axis"``
        (``"GT"``), and ``"col_axis"`` (``"Pred"``).

        Args:
            node (``Any``): The node the metric is attached to. When
                the object has a ``module`` attribute, as a
                `NodeWrapper` has, the method reads the class names
                from that attribute.
            name (str): Name of the metric. Unused.
            value (``Tensor``): The matrix, of shape ``[R, C]``.

        Returns:
            ``dict[str, Any]``: The matrix values and their labels.

        Example:
            >>> import torch
            >>> from types import SimpleNamespace
            >>> bar = LuxonisTQDMProgressBar()
            >>> node = SimpleNamespace(class_names=["cat", "dog"])
            >>> matrix = torch.tensor([[3, 1, 0], [0, 2, 1]])
            >>> info = bar.format_matrix_for_printing(node, "cm", matrix)
            >>> info["row_labels"], info["col_labels"]
            (['cat', 'dog'], ['cat', 'dog', 'no match'])
            >>> info["values"]
            [[3, 1, 0], [0, 2, 1]]
            >>> info["row_axis"], info["col_axis"]
            ('GT', 'Pred')

        """
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
    """Progress bar that prints plain text with ``tqdm``.

    `LuxonisModel` uses this bar when ``rich_logging`` is ``False`` in
    the config. The bar prints the results of an evaluation epoch as
    ``tabulate`` grids through the logger, so that the console and the
    log file receive the same text.

    """

    def __init__(self):
        """Initialize the bar with ``leave=True``.

        The finished train bar stays in the terminal at the end of each
        epoch, and the next epoch gets a new bar.

        """
        super().__init__(leave=True)

    @override
    def print_results(
        self,
        stage: str,
        loss: float,
        metrics: Mapping[str, Mapping[str, int | str | float]],
        matrices: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> None:
        """Print the results of an evaluation epoch through the logger.

        The output starts with a rule that holds the stage name, then
        the loss and a ``Metrics:`` heading. Each node in ``metrics``
        gets a rule with its name and a ``tabulate`` grid with the
        columns ``Name`` and ``Value``. The matrices of that node
        follow its grid. The matrices of the nodes that have no scalar
        metrics come last, under the title ``<node>/<matrix title>``.
        The title of a matrix is its name in title case, with spaces
        for underscores. A matrix prints as a rule with its title and
        a grid. The header of the grid holds the
        ``row_axis`` and ``col_axis`` names of the matrix and the
        column labels. Each row starts with its row label. A closing
        rule ends the output. Every line goes through ``logger.info``,
        so it reaches the console and the log file.

        Args:
            stage (str): Name of the stage, for example ``"Validation"``.
            loss (float): Mean loss of the epoch.
            metrics (``Mapping[str, Mapping[str, int | str | float]]``):
                Scalar metrics as ``{node_name: {metric_name: value}}``.
            matrices (``Mapping[str, Mapping[str, Mapping[str, Any]]]``):
                Matrix metrics as ``{node_name: {metric_name: matrix}}``,
                in the format of
                `BaseLuxonisProgressBar.format_matrix_for_printing`.

        """
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
        """Print one table as a ``tabulate`` grid through the logger.

        The output is a rule with ``title``, then the table in the
        ``fancy_grid`` format with right-aligned numbers.

        Args:
            title (str): Title of the table.
            table (``Iterable[tuple[str | int | float, ...]]``): The rows.
                Each row is a tuple with one value per column.
            column_names (list[str]): Names of the columns.

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
        """Start a new train bar and record the epoch start time.

        Lightning calls this hook at the start of every train epoch.
        The Lightning ``TQDMProgressBar`` base class creates a new bar,
        because ``leave`` is ``True``, and sets its description to
        ``Epoch N``. The start time feeds the duration in
        `LuxonisTQDMProgressBar.on_train_epoch_end`.

        Args:
            trainer (``pl.Trainer``): The trainer.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
        super().on_train_epoch_start(trainer, pl_module)
        self._epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Close the train bar and log the epoch summary.

        Lightning calls this hook at the end of every train epoch. The
        Lightning ``TQDMProgressBar`` base class sets the postfix of an
        enabled bar from `BaseLuxonisProgressBar.get_metrics` and closes
        the bar. Then one summary line goes to the log file only, in
        the form
        ``[Epoch <n>/<max>] Duration: <s>s | Train Loss: <loss>``. The
        duration is the time since
        `LuxonisTQDMProgressBar.on_train_epoch_start`, the validation
        of the epoch included. The loss is the ``train/loss`` value of
        ``trainer.callback_metrics``.

        Lightning runs this hook before
        `LuxonisLightningModule.on_train_epoch_end`, which logs
        ``train/loss``. So the line shows the value of the previous
        epoch. It shows ``N/A`` when the value is missing or zero, as
        in the first epoch.

        Args:
            trainer (``pl.Trainer``): The trainer.
            pl_module (LuxonisLightningModule): The model. The base
                class reads its metrics for the bar postfix.

        """
        super().on_train_epoch_end(trainer, pl_module)
        super()._log_progress(trainer)


@CALLBACKS.register()
class LuxonisRichProgressBar(RichProgressBar, BaseLuxonisProgressBar):
    """Progress bar that prints styled text with ``rich``.

    `LuxonisModel` uses this bar when ``rich_logging`` is ``True`` in
    the config. The bar prints the results of an evaluation epoch as
    ``rich`` tables on the console. It renders the same tables without
    terminal styling into a buffer and writes the buffer to the log file
    only.

    """

    def __init__(self):
        """Initialize the bar with ``leave=True`` and a log console.

        The finished train bar stays in the terminal at the end of each
        epoch. The log console writes into an in-memory buffer without
        terminal styling. `LuxonisRichProgressBar.print_results` writes
        the buffer to the log file and clears it.

        """
        super().__init__(leave=True)
        self._log_buffer = StringIO()
        self._log_console = Console(
            file=self._log_buffer, force_terminal=False
        )

    @property
    def console(self) -> Console:
        """The ``rich`` console that the bar prints to.

        The Lightning ``RichProgressBar`` base class creates the
        console when a stage starts and the bar is enabled. A bar that
        is disabled when the stage starts gets no console.

        Raises:
            RuntimeError: When the console does not exist yet. The
                message asks the user to set ``rich_logging`` to
                ``False`` in the config.

        """
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
        """Print the results of an evaluation epoch.

        On the console, the output starts with a magenta rule that
        holds the stage name, then the loss and a ``Metrics:`` heading.
        Each node in ``metrics`` gets a ``rich`` table with its name as
        the title and the columns ``Name`` and ``Value``. The matrices
        of that node follow its table. The matrices of the nodes that
        have no scalar metrics come last, under the title
        ``<node>/<matrix title>``. The title of a matrix is its name in
        title case, with spaces for underscores. A matrix prints as a
        table. Its header holds the ``row_axis`` and ``col_axis`` names
        of the matrix and the column labels. Each row starts with its
        row label. A closing rule ends the output. The same output,
        without terminal styling, goes to the log file only. The
        method then clears the log buffer.

        Args:
            stage (str): Name of the stage, for example ``"Validation"``.
            loss (float): Mean loss of the epoch.
            metrics (``Mapping[str, Mapping[str, int | str | float]]``):
                Scalar metrics as ``{node_name: {metric_name: value}}``.
            matrices (``Mapping[str, Mapping[str, Mapping[str, Any]]]``):
                Matrix metrics as ``{node_name: {metric_name: matrix}}``,
                in the format of
                `BaseLuxonisProgressBar.format_matrix_for_printing`.

        Raises:
            RuntimeError: When the console of the bar does not exist
                yet, see `LuxonisRichProgressBar.console`.

        """
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
        """Print one table as a ``rich`` table.

        The title is bold and the headers are bold magenta. The first
        column is magenta and the other columns are white. ``str``
        converts the first element of each row. A ``float`` in the
        other elements prints with five decimals. Any other element
        prints with ``str``.

        Args:
            title (str): Title of the table.
            table (``Iterable[tuple[str | int | float, ...]]``): The rows.
                Each row is a tuple with one value per column.
            column_names (list[str]): Names of the columns.
            console (``Console | None``): The console to print to.
                ``None`` means the console of the bar, the terminal.

        Raises:
            RuntimeError: When ``console`` is ``None`` and the console
                of the bar does not exist yet, see
                `LuxonisRichProgressBar.console`.

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
        """Start the train task and record the epoch start time.

        Lightning calls this hook at the start of every train epoch.
        The Lightning ``RichProgressBar`` base class adds a train task
        named after the epoch. From the second epoch on, it first stops
        the current display and starts a new one, because ``leave`` is
        ``True``. A disabled bar skips all of that. The start time
        feeds the duration in
        `LuxonisRichProgressBar.on_train_epoch_end`.

        Args:
            trainer (``pl.Trainer``): The trainer.
            pl_module (LuxonisLightningModule): The model. Unused.

        """
        super().on_train_epoch_start(trainer, pl_module)
        self._epoch_start_time = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: "lxt.LuxonisLightningModule"
    ) -> None:
        """Refresh the train task and log the epoch summary.

        Lightning calls this hook at the end of every train epoch. The
        Lightning ``RichProgressBar`` base class updates the metrics
        column of an enabled display and refreshes the display. The
        items come from `BaseLuxonisProgressBar.get_metrics`, with
        every tensor replaced by its ``item()`` value. Then one
        summary line goes to the log file only, in the form
        ``[Epoch <n>/<max>] Duration: <s>s | Train Loss: <loss>``. The
        duration is the time since
        `LuxonisRichProgressBar.on_train_epoch_start`, the validation
        of the epoch included. The loss is the ``train/loss`` value of
        ``trainer.callback_metrics``.

        Lightning runs this hook before
        `LuxonisLightningModule.on_train_epoch_end`, which logs
        ``train/loss``. So the line shows the value of the previous
        epoch. It shows ``N/A`` when the value is missing or zero, as
        in the first epoch.

        Args:
            trainer (``pl.Trainer``): The trainer.
            pl_module (LuxonisLightningModule): The model. The base
                class reads its metrics for the metrics column.

        """
        super().on_train_epoch_end(trainer, pl_module)
        super()._log_progress(trainer)


def build_optimizer_summary(
    optimizers: Sequence[Optimizer],
    schedulers: Sequence[LRSchedulerTypeUnion | LRSchedulerConfig],
    modules: Mapping[str, nn.Module],
) -> dict[str, Any]:
    """Build the summary of the optimizers and their parameter groups.

    The summary is a nested dictionary that `log_optimizer_summary`
    renders and writes as JSON to the log file. Each parameter group
    lists its hyperparameters and its *owners*, the modules of
    ``modules`` whose parameters it holds.

    Two denominators are in use:

    - The group-level percentages ``tensors_pct_of_model`` and
      ``params_pct_of_model`` are relative to all parameters of the
      model. The sum over all groups of all optimizers is 100% minus
      the share of the parameters that no group holds. A parameter
      that two groups hold counts twice in that sum.
    - The owner-level percentages ``tensors_pct_of_owner`` and
      ``params_pct_of_owner`` are relative to all parameters of that
      owner. Over all appearances of one owner they add up to 100%
      when every parameter of the owner is in exactly one group. This
      shows how the parameters of a node are split across groups.

    A frozen parameter counts in its group like any other parameter.
    The percentages include frozen parameters, and each group and
    owner reports its trainable and frozen counts separately.

    The owner ``"<external>"`` collects the parameters that a group
    holds but that no module in ``modules`` owns. A parameter that
    several modules share belongs to the first module of ``modules``
    that lists it. In every count, ``*_tensors`` is a number of
    parameter tensors and ``*_params`` is a number of elements. A
    percentage is ``0.0`` when its denominator is zero.

    The result has these keys:

    - ``n_optimizers``: Number of optimizers.
    - ``model_tensors``, ``model_params``: Totals over all owners,
      external ones included.
    - ``trainable_tensors``, ``trainable_params``, ``frozen_tensors``,
      ``frozen_params``: The totals split by ``requires_grad``.
    - ``optimizers``: One entry per optimizer with ``index``,
      ``optimizer`` and ``scheduler`` (class names), ``n_groups``, and
      ``groups``.

    Each group holds ``index``, ``n_tensors``, ``n_params``, the
    trainable and frozen counts, ``tensors_pct_of_model``,
    ``params_pct_of_model``, ``hyperparams``, and ``owners``.
    ``hyperparams`` holds the entries of the parameter group except
    ``params``, callables, lists, tuples, and dictionaries. ``owners``
    lists the owners in descending order of their ``n_params`` in the
    group. Each owner holds ``name``, ``n_tensors``,
    ``n_tensors_of_owner``, ``tensors_pct_of_owner``, ``n_params``,
    ``n_params_of_owner``, ``params_pct_of_owner``, and the trainable
    and frozen counts.

    Args:
        optimizers (``Sequence[Optimizer]``): The optimizers, in order.
        schedulers (``Sequence[LRSchedulerTypeUnion | LRSchedulerConfig]``):
            One entry per optimizer. A dictionary contributes the
            class name of its ``"scheduler"`` entry, as a Lightning
            scheduler config dictionary holds one. Any other object
            contributes its own class name, so ``None`` for an
            optimizer without a scheduler shows as ``"NoneType"``. A
            Lightning ``LRSchedulerConfig`` dataclass is not a
            dictionary, so it shows as ``"LRSchedulerConfig"``.
        modules (``Mapping[str, nn.Module]``): The owner modules keyed by
            name, usually the nodes of the model keyed by node name.

    Returns:
        ``dict[str, Any]``: The summary described above.

    Raises:
        ValueError: When ``optimizers`` and ``schedulers`` differ in
            length.

    Example:
        >>> from torch import nn
        >>> from torch.optim import SGD
        >>> from torch.optim.lr_scheduler import ConstantLR
        >>> backbone = nn.Linear(4, 8)
        >>> head = nn.Linear(8, 2)
        >>> optimizer = SGD(
        ...     [
        ...         {"params": backbone.parameters()},
        ...         {"params": head.parameters(), "lr": 0.1},
        ...     ],
        ...     lr=0.01,
        ... )
        >>> summary = build_optimizer_summary(
        ...     [optimizer],
        ...     [ConstantLR(optimizer, factor=1.0)],
        ...     {"backbone": backbone, "head": head},
        ... )
        >>> summary["n_optimizers"], summary["model_params"]
        (1, 58)
        >>> group = summary["optimizers"][0]["groups"][0]
        >>> group["hyperparams"]["lr"]
        0.01
        >>> round(group["params_pct_of_model"], 1)
        69.0
        >>> [owner["name"] for owner in group["owners"]]
        ['backbone']

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
    """Log the summary from `build_optimizer_summary`.

    With ``use_rich``, the summary goes to the global ``rich`` console
    as nested panels. A header panel holds the totals. Then one panel
    per optimizer holds one panel per group. A group panel holds one
    panel with the hyperparameters and one with the owners. Without
    ``use_rich``, the summary goes through ``logger.info`` as an
    indented plain-text list, which reaches the console and the log
    file. In both cases the function also writes the summary as JSON
    to the log file only, with ``str`` for a value that JSON cannot
    encode.

    Args:
        summary (``dict[str, Any]``): The summary from
            `build_optimizer_summary`.
        use_rich (bool): Whether to render the summary with ``rich``
            panels.

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
    """Parameter tallies per owner, with each parameter counted once."""

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
