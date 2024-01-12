from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(help="Luxonis Train CLI", add_completion=False)


class View(str, Enum):
    train = "train"
    val = "val"
    test = "test"

    def __str__(self):
        return self.value


ConfigType = Annotated[
    Optional[Path],
    typer.Option(
        help="Path to the configuration file.",
        show_default=False,
    ),
]

OptsType = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

ViewType = Annotated[View, typer.Option(help="Which dataset view to use.")]

SaveDirType = Annotated[
    Optional[Path],
    typer.Option(help="Where to save the inference results."),
]


@app.command()
def train(config: ConfigType = None, opts: OptsType = None):
    """Start training."""
    from luxonis_train.core import Trainer

    Trainer(str(config), opts).train()


@app.command()
def eval(config: ConfigType = None, view: ViewType = View.val, opts: OptsType = None):
    """Evaluate model."""
    from luxonis_train.core import Trainer

    Trainer(str(config), opts).test(view=view.name)


@app.command()
def tune(config: ConfigType = None, opts: OptsType = None):
    """Start hyperparameter tuning."""
    from luxonis_train.core import Tuner

    Tuner(str(config), opts).tune()


@app.command()
def export(config: ConfigType = None, opts: OptsType = None):
    """Export model."""
    from luxonis_train.core import Exporter

    Exporter(str(config), opts).export()


@app.command()
def infer(
    config: ConfigType = None,
    view: ViewType = View.val,
    save_dir: SaveDirType = None,
    opts: OptsType = None,
):
    """Run inference."""
    from luxonis_train.core import Inferer

    Inferer(str(config), opts, view=view.name, save_dir=save_dir).infer()


def version_callback(value: bool):
    if value:
        typer.echo(f"LuxonisTrain Version: {version(__package__)}")
        raise typer.Exit()


@app.callback()
def common(
    _: Annotated[
        bool,
        typer.Option(
            "--version", callback=version_callback, help="Show version and exit."
        ),
    ] = False,
):
    ...


if __name__ == "__main__":
    app()
