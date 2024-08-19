import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from luxonis_ml.enums import SplitType

app = typer.Typer(
    help="Luxonis Train CLI",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


ConfigType = Annotated[
    Optional[str],
    typer.Option(
        help="Path to the configuration file.",
        show_default=False,
        metavar="FILE",
    ),
]

OptsType = Annotated[
    Optional[list[str]],
    typer.Argument(
        help="A list of optional CLI overrides of the config file.",
        show_default=False,
    ),
]

ViewType = Annotated[SplitType, typer.Option(help="Which dataset view to use.")]

SaveDirType = Annotated[
    Optional[Path],
    typer.Option(help="Where to save the inference results."),
]


@app.command()
def train(
    config: ConfigType = None,
    resume: Annotated[
        Optional[str], typer.Option(help="Resume training from this checkpoint.")
    ] = None,
    opts: OptsType = None,
):
    """Start training."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).train(resume_weights=resume)


@app.command()
def test(
    config: ConfigType = None, view: ViewType = SplitType.VAL, opts: OptsType = None
):
    """Evaluate model."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).test(view=view.value)


@app.command()
def tune(config: ConfigType = None, opts: OptsType = None):
    """Start hyperparameter tuning."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).tune()


@app.command()
def export(config: ConfigType = None, opts: OptsType = None):
    """Export model."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).export()


@app.command()
def infer(
    config: ConfigType = None,
    view: ViewType = SplitType.VAL,
    save_dir: SaveDirType = None,
    opts: OptsType = None,
):
    """Run inference."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(config, opts).infer(view=view.value, save_dir=save_dir)


@app.command()
def inspect(
    config: ConfigType = None,
    view: Annotated[
        SplitType,
        typer.Option(
            ...,
            "--view",
            "-v",
            help="Which split of the dataset to inspect.",
            case_sensitive=False,
        ),
    ] = "train",  # type: ignore
    opts: OptsType = None,
):
    """Inspect dataset."""
    from lightning.pytorch import seed_everything
    from luxonis_ml.data.__main__ import inspect as lxml_inspect

    from luxonis_train.utils.config import Config

    cfg = Config.get_config(config, opts)
    if cfg.trainer.seed is not None:
        seed_everything(cfg.trainer.seed, workers=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.dump(
            [
                a.model_dump()
                for a in cfg.trainer.preprocessing.get_active_augmentations()
                if a.name != "Normalize"
            ],
            f,
        )

        if "dataset_name" not in cfg.loader.params:
            raise ValueError("dataset_name is not set in the config")

        lxml_inspect(
            name=cfg.loader.params["dataset_name"],
            view=view,
            aug_config=f.name,
        )


@app.command()
def archive(
    executable: Annotated[
        str,
        typer.Option(
            help="Path to the model file.", show_default=False, metavar="FILE"
        ),
    ],
    config: ConfigType = None,
    opts: OptsType = None,
):
    """Generate NN archive."""
    from luxonis_train.core import LuxonisModel

    LuxonisModel(str(config), opts).archive(executable)


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
    source: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to a python file with custom components. "
            "Will be sourced before running the command.",
            metavar="FILE",
        ),
    ] = None,
):
    if source:
        exec(source.read_text(), globals(), globals())


if __name__ == "__main__":
    app()
