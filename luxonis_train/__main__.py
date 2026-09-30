"""The ``luxonis_train`` command line interface.

The training, evaluation, export, and annotation commands build a
`LuxonisModel <luxonis_train.core.core.LuxonisModel>` from a config and
call one of its methods. ``inspect`` only reads the loaders of the
model. ``--model`` and ``--variant`` select a packaged config, so
``--config`` is optional. ``list-models`` and ``info`` describe the
packaged models. The ``upgrade`` group migrates a config, a checkpoint,
or the installation.

The global ``--source`` option runs one or more Python files with custom
components before the command, so that the components register.

"""

import importlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, cast

import yaml
from cyclopts import App, Group, Parameter, validators
from loguru import logger
from luxonis_ml.typing import Params, PathType

from luxonis_train.upgrade import upgrade_config, upgrade_installation

_SECTION_BY_PACKAGE = {
    "backbones": "Backbone",
    "necks": "Neck",
    "heads": "Head",
}

if TYPE_CHECKING:
    import numpy as np
    from rich.console import Console

    from luxonis_train import LuxonisModel
    from luxonis_train.config import NodeConfig
    from luxonis_train.config.predefined_models import BasePredefinedModel
    from luxonis_train.loaders import BaseLoaderTorch

    OptsType: TypeAlias = list[str] | None
    LauncherToken: TypeAlias = str
    LauncherSource: TypeAlias = list[Path] | None
else:
    OptsType = Annotated[
        list[str] | None, Parameter(json_list=False, json_dict=False)
    ]
    LauncherToken = Annotated[
        str, Parameter(show=False, allow_leading_hyphen=True)
    ]
    LauncherSource = Annotated[
        list[Path] | None,
        Parameter(
            help="Path to a python module with custom components. "
            "This module will be sourced before running a command."
        ),
    ]


app = App(
    help="Luxonis Train CLI",
    version=lambda: f"LuxonisTrain v{version('luxonis_train')}",
)
app.meta.group_parameters = Group("Global Parameters", sort_key=0)
app["--help"].group = app.meta.group_parameters
app["--version"].group = app.meta.group_parameters

upgrade_app = app.command(App(name="upgrade"))

training_group = Group.create_ordered("Training")
evaluation_group = Group.create_ordered("Evaluation")
export_group = Group.create_ordered("Export")
annotation_group = Group.create_ordered("Annotation")
management_group = Group.create_ordered("Management")


def create_model(
    config: PathType | Params | None = None,
    opts: list[str] | None = None,
    weights: PathType | None = None,
    allow_empty_dataset: bool = False,
    *,
    model: str | None = None,
    variant: str | None = None,
) -> "LuxonisModel":
    """Build the model that a CLI command operates on.

    The CLI imports ``luxonis_train`` without its submodules, so that
    it starts fast. This function reloads the package, which runs the
    full imports, and then builds a `LuxonisModel
    <luxonis_train.core.core.LuxonisModel>`.

    Args:
        config (``PathType | Params | None``): Path or URL of the config
            file, or the config as a dictionary. With ``None``, the
            packaged config of ``model`` applies, else the config stored
            in the ``weights`` checkpoint. Without those, ``opts`` alone
            builds the config from the defaults, and `Config.get_config`
            raises ``ValueError`` when ``opts`` is ``None`` too.
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``["trainer.epochs", "10"]``.
        weights (``PathType | None``): Path or URL of a checkpoint. Its
            dataset metadata applies, and its weights load as soon as
            the model is built.
        allow_empty_dataset (bool): When ``True``, a `DummyLoader`
            replaces a loader that fails to initialize, so the model
            builds without a dataset.
        model (str | None): Name of a packaged predefined model, with
            an optional version suffix such as ``"detection:v1"``.
            Mutually exclusive with ``config``.
        variant (str | None): Variant of the packaged model. Requires
            ``model``.

    Returns:
        LuxonisModel: The model, with its loaders and its trainer built.

    """
    importlib.reload(sys.modules["luxonis_train"])

    from luxonis_train import LuxonisModel

    return LuxonisModel(
        config,
        opts,
        model=model,
        variant=variant,
        weights=weights,
        allow_empty_dataset=allow_empty_dataset,
    )


@app.command(group=training_group, sort_key=1)
def train(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
    debug: bool = False,
):
    """Train the model.

    The run writes its log, its config, and its checkpoints to the run
    save directory under ``tracker.save_directory``.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        weights (str | None): Path or URL of a checkpoint. With
            ``trainer.resume_training`` set in the config, the run
            continues from it with its optimizer, its scheduler, and its
            epoch count. Otherwise only the weights load.
        debug (bool): When ``True``, a `DummyLoader` replaces a loader
            that fails to initialize, so the model builds without a
            valid dataset.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=debug,
        model=model,
        variant=variant,
    ).train(weights=weights)


@app.command(group=training_group, sort_key=2)
def tune(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
    debug: bool = False,
):
    """Search the hyperparameters with Optuna.

    The ``tuner`` section of the config defines the study. Each trial
    trains the model with sampled values. The trials go to
    ``tuner_study.csv`` in the run save directory, and the best
    parameters go to the parent tracker of the study.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        weights (str | None): Path or URL of a checkpoint. It supplies
            the config when ``config`` and ``model`` are omitted. The
            trials do not load its weights.
        debug (bool): When ``True``, a `DummyLoader` replaces a loader
            that fails to initialize, so the model builds without a
            valid dataset.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=debug,
        model=model,
        variant=variant,
    ).tune()


@app.command(group=training_group, sort_key=3)
def inspect(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    view: Literal["train", "val", "test"] = "train",
    size_multiplier: Annotated[
        float, Parameter(["--size_multiplier", "-s"])
    ] = 1.0,
    list_augmentations: bool = False,
):
    """Show the samples of a dataset view with their labels drawn.

    A window named ``Visualization`` shows one sample at a time. Any
    key moves to the next sample, and ``q`` or ``Esc`` closes the
    window. The loader does not normalize the images.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the defaults.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        view (``Literal["train", "val", "test"]``): The dataset view to
            inspect.
        size_multiplier (float): Scale factor for the image the loader
            returns. ``1.0`` keeps its size. The flags are
            ``--size_multiplier`` and ``-s``.
        list_augmentations (bool): When ``True``, a footer lists the
            augmentations applied to the sample. The footer reads
            ``Augmentations: none`` when the sample has no tracked
            augmentation. A loader that does not wrap a ``luxonis_ml``
            loader never has one.

    """
    import cv2

    @lru_cache
    def get_window() -> str:
        window_name = "Visualization"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        return window_name

    for viz in _yield_visualizations(
        config=config,
        view=view,
        size_multiplier=size_multiplier,
        list_augmentations=list_augmentations,
        opts=opts,
        model=model,
        variant=variant,
    ):
        window_name = get_window()
        cv2.resizeWindow(window_name, width=viz.shape[1], height=viz.shape[0])
        cv2.imshow(window_name, viz)
        if cv2.waitKey() in {ord("q"), 27}:
            break
    cv2.destroyAllWindows()


@app.command(group=evaluation_group, sort_key=1)
def test(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    view: Literal["train", "val", "test"] = "test",
    weights: str | None = None,
    debug: bool = False,
):
    """Evaluate the model on a dataset view.

    The command runs the test loop of the trainer on ``view`` and
    finalizes the tracked run.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        view (``Literal["train", "val", "test"]``): The dataset view to
            evaluate.
        weights (str | None): Path or URL of the checkpoint to evaluate. If
            omitted, ``model.weights`` of the config applies.
        debug (bool): When ``True``, a `DummyLoader` replaces a loader
            that fails to initialize, so the model builds without a
            valid dataset.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=debug,
        model=model,
        variant=variant,
    ).test(view=view, weights=weights)


@app.command(group=evaluation_group, sort_key=2)
def infer(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    view: Literal["train", "val", "test"] = "val",
    save_dir: Path | None = None,
    source_path: str | None = None,
    weights: str | None = None,
):
    """Run inference and show or save the visualizations.

    Without ``source_path``, the model runs over the dataset view.
    With it, the model runs over one image, over every image of a
    directory, or over every frame of a video. Without ``save_dir``,
    each visualizer opens a window named ``<node>/<visualizer>``, and
    ``q`` or ``Esc`` stops the run. An image or a directory passes
    through a temporary dataset named ``infer_from_directory``. The
    command deletes an existing local dataset of that name first, and
    deletes the temporary one at the end. A `DummyLoader` replaces a
    loader that fails to initialize, so an image or a directory source
    runs without a dataset. A video source needs a real validation
    loader, because each frame passes through its ``augment_test_image``
    method. A `DummyLoader` does not implement that method.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        view (``Literal["train", "val", "test"]``): The dataset view to
            use when ``source_path`` is omitted.
        save_dir (``Path | None``): Directory for the visualizations,
            as ``.png`` images or as ``.mp4`` videos. The command
            creates it when it is missing.
        source_path (str | None): Path to an image file, to a directory
            of images, or to a video file. A directory contributes the
            files directly inside it with an image extension, such as
            ``.jpg`` or ``.png``. The extension identifies a video:
            ``.mp4``, ``.mov``, ``.avi``, ``.mkv``, or ``.webm``.
        weights (str | None): Path or URL of the checkpoint to run. If
            omitted, ``model.weights`` of the config applies.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).infer(
        view=view,
        save_dir=save_dir,
        source_path=source_path,
        weights=weights,
    )


@app.command(group=annotation_group, sort_key=0)
def annotate(
    opts: OptsType = None,
    /,
    *,
    dir_path: Path,
    dataset_name: str,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
    bucket_storage: Literal["local", "gcs"] = "local",
    delete_local: bool = True,
    delete_remote: bool = True,
    team_id: str | None = None,
):
    """Annotate a directory of images into a new dataset.

    The model predicts on every file directly inside ``dir_path`` with
    an image extension, such as ``.jpg`` or ``.png``. Each head turns
    its predictions into annotations. The command writes them into a
    ``LuxonisDataset`` named ``dataset_name``, creates its splits when
    it is not empty, and prints its summary. The images pass through a
    temporary dataset named ``infer_from_directory``. The command
    deletes an existing local dataset of that name first, and deletes
    the temporary one at the end. A `DummyLoader` replaces a loader
    that fails to initialize, so the command runs without a dataset.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        dir_path (``Path``): Directory with the images to annotate. It
            must be a directory.
        dataset_name (str): Name of the dataset to create.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        weights (str | None): Path or URL of the checkpoint to annotate with.
            If omitted, ``model.weights`` of the config applies.
        bucket_storage (``Literal["local", "gcs"]``): Where the
            command stores the new dataset.
        delete_local (bool): Delete an existing local dataset of the
            same name first. With ``False``, the command adds the
            annotations to the existing dataset.
        delete_remote (bool): Also delete the remote copy of an
            existing dataset of the same name.
        team_id (str | None): Team that owns the dataset. ``None``
            reads ``LUXONISML_TEAM_ID`` from the environment.

    """
    lx_model = create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    )

    lx_model.annotate(
        dir_path=dir_path,
        dataset_name=dataset_name,
        weights=weights,
        bucket_storage=bucket_storage,
        delete_local=delete_local,
        delete_remote=delete_remote,
        team_id=team_id,
    )


@app.command(group=export_group, sort_key=1)
def export(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    save_path: str | None = None,
    weights: str | None = None,
    ckpt_only: bool = False,
):
    """Export the model to ONNX.

    The command writes ``<name>.onnx``. For a model with one input,
    it also writes a ``<name>.yaml`` config for ``modelconverter``
    next to it. ``<name>`` is ``exporter.name`` or ``model.name``. With
    ``--ckpt-only``, the command writes only ``<name>.ckpt``. It saves
    the checkpoint again with the current config, execution order, and
    dataset metadata, which refreshes an older checkpoint. Without
    ``--ckpt-only``, the command uploads the files to the run when
    ``exporter.upload_to_run`` is set, and to ``exporter.upload_url``
    when that is set. A `DummyLoader` replaces a loader that fails to
    initialize, so the command runs without a dataset.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        save_path (str | None): Directory for the exported files, or a
            file path whose stem names them. If omitted, the files go
            to the ``export`` directory of the run save directory.
        weights (str | None): Path or URL of the checkpoint to export. If
            omitted, ``model.weights`` of the config applies.
        ckpt_only (bool): When ``True``, write only the ``.ckpt`` file.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).export(save_path=save_path, weights=weights, ckpt_only=ckpt_only)


@app.command(group=export_group, sort_key=2)
def archive(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    executable: str | None = None,
    weights: str | None = None,
):
    """Pack the exported model into an NN Archive.

    The archive holds the executable, its ``.data`` file when one
    exists, and a config with the inputs, the outputs, the
    preprocessing, and the heads. It goes to the ``archive`` directory
    of the run save directory. The command uploads the archive to
    ``archiver.upload_url`` when that is set, and to the run when
    ``archiver.upload_to_run`` is set. A `DummyLoader` replaces a
    loader that fails to initialize, so the command runs without a
    dataset.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        executable (str | None): Path to the exported ONNX model.
            Another format stops the command with a
            ``NotImplementedError``. If omitted, the command exports
            the model to ONNX first.
        weights (str | None): Path or URL of the checkpoint to archive. If
            omitted, ``model.weights`` of the config applies.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).archive(path=executable, weights=weights)


@app.command(group=export_group, sort_key=3)
def convert(
    opts: OptsType = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    save_dir: str | None = None,
    weights: str | None = None,
):
    """Export, archive, and convert the model for a target platform.

    The command exports the model to ONNX, packs it into an NN Archive,
    and then runs the conversions the config activates:

    - ``exporter.blobconverter.active``: a ``.blob`` for RVC2 through
      ``blobconverter``, which is deprecated.
    - ``exporter.hubai.active``: an NN Archive for
      ``exporter.hubai.platform`` (``rvc2``, ``rvc3``, or ``rvc4``)
      through the HubAI SDK.

    The command skips a conversion whose package is not installed and
    logs it. A `DummyLoader` replaces a loader that fails to
    initialize, so the command runs without a dataset.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        save_dir (str | None): Directory for every output. If omitted,
            the outputs go under the run save directory.
        weights (str | None): Path or URL of the checkpoint to convert. If
            omitted, ``model.weights`` of the config applies.

    """
    create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=True,
        model=model,
        variant=variant,
    ).convert(save_dir=save_dir, weights=weights)


@app.command(group=export_group, sort_key=1)
def quantize(
    opts: list[str] | None = None,
    /,
    *,
    config: str | None = None,
    model: str | None = None,
    variant: str | None = None,
    weights: str | None = None,
):
    """Quantize the model with AIMET.

    The ``exporter.aimet`` section of the config sets every option.
    The command evaluates the float model on the validation view. It
    runs post-training quantization, calibrated on the validation
    images, and evaluates the result. It then runs quantization-aware
    training on the train view and evaluates again. It writes the
    quantized ONNX and its NN Archive to the ``aimet`` directory of
    the run save directory. The command needs a dataset and the
    ``aimet`` extra.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        config (str | None): Path or URL of the config file. Mutually
            exclusive with ``model``. If omitted, the packaged config of
            ``model`` applies, else the config stored in the ``weights``
            checkpoint.
        model (str | None): Name of a packaged predefined model, for
            example ``"detection"`` or ``"detection:v1"``. Run
            ``luxonis_train list-models`` to see the options.
        variant (str | None): Variant of the predefined model, for
            example ``"light"`` or ``"heavy"``. Defaults to the default
            variant of the model.
        weights (str | None): Path or URL of the checkpoint to quantize. If
            omitted, ``model.weights`` of the config applies.

    """
    lx_model = create_model(
        config,
        opts,
        weights=weights,
        allow_empty_dataset=False,
        model=model,
        variant=variant,
    )
    lx_model.quantize()


@app.command(group=management_group, sort_key=1, name="list-models")
def list_models():
    """List the packaged predefined models.

    The command prints a table with one row per model, or a notice when
    no packaged model exists. The ``Variants`` column marks the default
    variant with ``*`` and shows ``<default>`` for a config without a
    variant name. The ``Versions`` column marks the latest version with
    ``*`` and shows ``-`` when the config names no registered model
    class. The marked values apply when the option is omitted.

    """
    from rich import box
    from rich.console import Console
    from rich.table import Table

    from luxonis_train.config.predefined import list_predefined_models

    entries = list_predefined_models()
    if not entries:
        Console().print("[yellow]No packaged predefined models found.[/]")
        return

    table = Table(
        title="Packaged predefined models",
        caption="[dim]* default when the option is omitted[/]",
        box=box.ROUNDED,
    )
    table.add_column("Model", style="bold cyan")
    table.add_column("Variants")
    table.add_column("Versions", style="green")
    for name, file_variants in entries.items():
        table.add_row(
            name, _variants_cell(name, file_variants[0]), _versions_cell(name)
        )

    Console().print(table)


@app.command(group=management_group, sort_key=2)
def info(*, model: str, variant: str | None = None):
    """Print the documentation of a packaged predefined model.

    The first panel names the model, the variant, and the registry key
    of the resolved model class. It shows the docstring of the class,
    or a fallback sentence when the class has none. One panel per
    component follows. A simple model gets the backbone, the neck when
    it has one, and the head. Any other model gets every node. Each
    panel names the node class and its variant. It shows the docstring
    of the class, of its ``__init__`` when the class has none, or a
    notice when both are missing. An unknown model, variant, or
    version stops the command with a ``ValueError``.

    Args:
        model (str): Name of a packaged model, with an optional version
            suffix, for example ``"detection:v1"`` or
            ``"detection:latest"``.
        variant (str | None): The variant to describe. Defaults to the
            default variant of the model.

    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    from luxonis_train.config.predefined import (
        parse_model_spec,
        resolve_predefined_config,
    )
    from luxonis_train.config.predefined_versions import (
        resolve_predefined_class,
        resolved_class_name,
    )

    importlib.import_module("luxonis_train.nodes")
    importlib.import_module("luxonis_train.config.predefined_models")

    family, requested_version = parse_model_spec(model)
    config = yaml.safe_load(
        resolve_predefined_config(family, variant).path.read_text()
    )
    predefined_config = config["model"]["predefined_model"]
    class_family = predefined_config["name"]
    version: int | str
    if requested_version is None or requested_version == "latest":
        version = "latest"
    else:
        version = int(requested_version)
    model_class = resolve_predefined_class(class_family, version)
    params = dict(predefined_config.get("params") or {})
    # The config layer allows `variant` both at the `predefined_model`
    # level and inside `params` (where it takes precedence).
    params_variant = params.pop("variant", None)
    selected_variant = (
        variant
        or params_variant
        or predefined_config.get("variant", "default")
    )
    predefined_model = cast(Any, model_class)(
        variant=selected_variant,
        **params,
    )
    resolved_name = resolved_class_name(class_family, version)

    console = Console()
    description = _docstring_to_text(model_class.__dict__.get("__doc__"))
    if not description:
        description = f"Predefined {class_family} architecture."
    console.print(
        Panel(
            Text(description),
            title=f"[bold]{family}[/] · {selected_variant} · {resolved_name}",
            border_style="cyan",
        )
    )

    node_configs = {node.name: node for node in predefined_model.nodes}
    for section, node_name in _info_components(predefined_model):
        if node_name is None:
            continue
        _print_node_panel(console, section, node_name, node_configs[node_name])


@upgrade_app.command()
def config(
    config: Annotated[
        Path,
        Parameter(validator=validators.Path(exists=True)),
        Parameter(validator=validators.Path(ext={"yaml", "yml", "json"})),
    ],
    output: Annotated[
        Path | None,
        Parameter(validator=validators.Path(ext={"yaml", "yml", "json"})),
    ] = None,
):
    """Upgrade a config file to the current schema.

    The command reads the file as JSON when its suffix is ``.json``
    and as YAML otherwise. It applies the migration steps of
    `upgrade_config` and writes the result in the format of the
    output suffix. When the file is already current, the command
    writes it back without migration. It drops a deprecated
    ``config_version`` field in both cases.

    Args:
        config (``Path``): Path to the config file to upgrade. It must
            exist and end with ``.yaml``, ``.yml``, or ``.json``.
        output (``Path | None``): Where to write the upgraded config.
            It must end with ``.yaml``, ``.yml``, or ``.json``. If
            omitted, the command overwrites ``config``.

    """
    new_cfg = upgrade_config(config)

    output = output or config
    if output.suffix == ".json":
        output.write_text(json.dumps(new_cfg, indent=2))
    else:
        with open(output, "w") as f:
            yaml.safe_dump(
                new_cfg, f, sort_keys=False, default_flow_style=False
            )


@upgrade_app.command(name=["checkpoint", "ckpt"])
def checkpoint(
    opts: OptsType = None,
    /,
    *,
    path: Annotated[
        Path,
        Parameter(validator=validators.Path(exists=True)),
    ],
    output: Path | None = None,
    config: Path | None = None,
):
    """Upgrade a checkpoint file to the current format.

    The command builds the model from ``config``, or from the config
    stored in the checkpoint, and loads the checkpoint into it. Then
    it runs the validation loop once to attach the model to the
    trainer. It saves the checkpoint again with the current config,
    execution order, and dataset metadata. A checkpoint without a
    stored config needs ``config``. Without a dataset, a `DummyLoader`
    stands in for the validation loop. The command is also named
    ``ckpt``.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens, for example ``trainer.epochs 10``.
        path (``Path``): Path to the checkpoint. It must exist.
        output (``Path | None``): Where to write the upgraded
            checkpoint. If omitted, the command overwrites ``path``.
        config (``Path | None``): Config file to build the model from.
            If omitted, the config stored in the checkpoint applies.

    """
    from luxonis_train import LuxonisModel

    logger.info("Performing a full checkpoint upgrade.")
    model = LuxonisModel(config, opts, weights=path, allow_empty_dataset=True)
    model.lightning_module.load_checkpoint(path)

    # The validation run attaches the module to the trainer.
    model.pl_trainer.validate(
        model.lightning_module,
        model.pytorch_loaders["val"],
        verbose=False,
    )
    model.pl_trainer.save_checkpoint(output or path, weights_only=False)
    logger.info(f"Saved upgraded checkpoint to '{output}'")


@upgrade_app.default()
def upgrade():
    """Upgrade the ``luxonis-train`` installation.

    Without a subcommand, the command reads the latest release from
    PyPI. When it differs from the installed version, the command
    upgrades ``pip``, ``luxonis_train``, and ``luxonis_ml[data]`` with
    pip. When the check fails, the command logs a message and stops. The
    ``config`` and ``checkpoint`` subcommands upgrade user files
    instead.

    """
    upgrade_installation()


@app.meta.default
def launcher(
    *tokens: LauncherToken,
    source: LauncherSource = None,
):
    """Run the custom component files, then run the command.

    This is the entry point of the CLI. The launcher executes each
    file in ``source`` as a module before the command runs. The custom
    nodes, losses, and other components in those files then register.
    The launcher skips a file that does not resolve to a module. With
    ``--source`` on the command line, ``luxonis_train`` imports every
    submodule at startup, so the files can import from it.

    Args:
        *tokens (str): The command and its arguments, passed on
            unchanged.
        source (``list[Path] | None``): Python files with custom
            components. Give ``--source`` once per file.

    """
    if source:
        for src in source:
            spec = importlib.util.spec_from_file_location(src.stem, src)
            if spec:
                module = importlib.util.module_from_spec(spec=spec)
                if spec.loader:
                    spec.loader.exec_module(module)
    app(tokens)


def _get_visualization_item(
    loader: "BaseLoaderTorch", index: int
) -> tuple[dict[str, "np.ndarray"], dict[str, "np.ndarray"], list[str]]:
    """Return one sample of a loader as NumPy arrays.

    For a loader that wraps a ``luxonis_ml`` loader, the sample comes
    from that inner loader. The images and the labels stay NumPy
    arrays. The function remaps the keypoints when the loader has a
    keypoint mapping, and lists the tracked augmentations. It reads
    any other loader directly: the image tensors become ``[H, W, C]``
    arrays, the label tensors become arrays, and the augmentation
    list is empty.

    Args:
        loader (BaseLoaderTorch): The loader to read from.
        index (int): The index of the sample.

    Returns:
        ``tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]``:
        The images by source name, the labels by task, and the names of
        the tracked augmentations.

    """
    import numpy as np
    from luxonis_ml.data.utils.cli_utils import get_tracked_augmentations

    raw_loader = getattr(loader, "loader", None)
    if raw_loader is not None:
        sample = raw_loader[index]
        images, labels = sample
        if isinstance(images, np.ndarray):
            images = {loader.image_source: images}
        remap_keypoints = getattr(loader, "_remap_keypoints", None)
        if (
            getattr(loader, "kpts_mapping_per_task", None) is not None
            and remap_keypoints is not None
        ):
            labels = remap_keypoints(labels)
        return (
            images,
            labels,
            list(get_tracked_augmentations(sample.metadata) or {}),
        )

    images, labels = loader[index]
    if not isinstance(images, dict):
        images = {loader.image_source: images}
    return (
        {
            name: image.numpy().transpose(1, 2, 0)
            for name, image in images.items()
        },
        {task: label.numpy() for task, label in labels.items()},
        [],
    )


def _yield_visualizations(
    opts: OptsType = None,
    config: str | None = None,
    view: Literal["train", "val", "test"] = "train",
    size_multiplier: Annotated[
        float, Parameter(["--size_multiplier", "-s"])
    ] = 1.0,
    list_augmentations: bool = False,
    *,
    model: str | None = None,
    variant: str | None = None,
) -> Iterator["np.ndarray"]:
    """Yield the rendered samples of a dataset view, one at a time.

    The generator appends an override that disables normalization to
    ``opts``, which mutates the list in place. It then builds the
    model and walks the loader of ``view``. For each sample, it
    converts the main image from RGB to BGR, decodes the text metadata
    labels, scales the image by ``size_multiplier``, and draws the
    labels.

    Args:
        opts (list[str] | None): Config overrides as alternating key
            and value tokens.
        config (str | None): Path or URL of the config file.
        view (``Literal["train", "val", "test"]``): The dataset view.
        size_multiplier (float): Scale factor for the image.
        list_augmentations (bool): When ``True``, a footer lists the
            tracked augmentations.
        model (str | None): Name of a packaged predefined model.
        variant (str | None): Variant of the predefined model.

    Yields:
        ``np.ndarray``: The rendered sample in BGR, of shape
        ``[H, W, 3]``. It is a labeled grid with the image and one
        panel per task with drawn labels. The classification and the
        metadata labels follow as text. With ``list_augmentations``,
        the augmentation footer comes last.

    """
    import cv2
    import numpy as np
    from luxonis_ml.data.utils.visualizations import (
        add_augmentation_footer,
        visualize,
    )

    from luxonis_train.utils.general import decode_text_metadata_labels

    opts = opts or []
    opts.extend(["trainer.preprocessing.normalize.active", "False"])

    lx_model = create_model(config, opts, model=model, variant=variant)

    loader = lx_model.loaders[view]

    metadata_types = loader.get_metadata_types()
    categorical_encodings = loader.get_categorical_encodings()
    for idx in range(len(loader)):
        np_images, np_labels, augmentations = _get_visualization_item(
            loader, idx
        )
        main_image = np_images[loader.image_source]
        main_image = cv2.cvtColor(main_image, cv2.COLOR_RGB2BGR).astype(
            np.uint8
        )
        np_labels = decode_text_metadata_labels(np_labels, metadata_types)

        h, w, _ = main_image.shape
        new_h, new_w = int(h * size_multiplier), int(w * size_multiplier)
        main_image = cv2.resize(main_image, (new_w, new_h))
        viz = visualize(
            image=main_image,
            labels=np_labels,
            classes=loader.get_classes(),
            source_name=loader.image_source,
            categorical_encodings=categorical_encodings,
        )
        if list_augmentations:
            viz = add_augmentation_footer(viz, augmentations)
        yield viz


def _variants_cell(name: str, default: str | None) -> str:
    from luxonis_train.config.predefined import list_variants

    labels = []
    for v in list_variants(name):
        label = v if v is not None else "<default>"
        labels.append(f"{label}*" if v == default else label)
    return ", ".join(labels)


def _versions_cell(name: str) -> str:
    from luxonis_train.config.predefined import class_family
    from luxonis_train.config.predefined_versions import list_versions

    family = class_family(name)
    versions = list_versions(family) if family else {}
    if not versions:
        return "-"
    latest = max(versions)
    return ", ".join(f"v{v}*" if v == latest else f"v{v}" for v in versions)


def _info_components(
    predefined_model: "BasePredefinedModel",
) -> tuple[tuple[str, str | None], ...]:
    from luxonis_train.config.predefined_models.base_predefined_model import (
        SimplePredefinedModel,
    )

    if isinstance(predefined_model, SimplePredefinedModel):
        return (
            ("Backbone", predefined_model._backbone),
            (
                "Neck",
                predefined_model._neck if predefined_model._use_neck else None,
            ),
            ("Head", predefined_model._head),
        )
    return tuple(
        (_node_section(node.name), node.name)
        for node in predefined_model.nodes
    )


def _node_section(node_name: str) -> str:
    from luxonis_train.registry import NODES

    module = NODES.get(node_name).__module__
    for package, label in _SECTION_BY_PACKAGE.items():
        if f".nodes.{package}." in module:
            return label
    return "Node"


def _print_node_panel(
    console: "Console",
    section: str,
    node_name: str,
    node_config: "NodeConfig",
) -> None:
    from rich.panel import Panel
    from rich.text import Text

    from luxonis_train.registry import NODES

    node_class = NODES.get(node_name)
    node_doc = (
        node_class.__dict__.get("__doc__") or node_class.__init__.__doc__
    )
    body = Text(_docstring_to_text(node_doc) or "No documentation available.")
    variant_label = node_config.variant or "default"
    console.print(
        Panel(
            body,
            title=f"[bold]{section}[/] · {node_name} ({variant_label})",
            border_style="green",
        )
    )


def _docstring_to_text(doc: str | None) -> str:
    """Flatten the reST markup of a docstring for terminal output.

    The function dedents the text with ``inspect.cleandoc``. It
    removes the comment and directive lines, turns an external link
    into ``text (url)``, and strips the roles and the backticks. The
    body of a ``code-block`` stays, because it follows the directive
    indented.

    Args:
        doc (str | None): The raw docstring. ``None`` counts as empty.

    Returns:
        str: The plain text.

    """
    import inspect
    import re

    text = inspect.cleandoc(doc or "")
    # A reST comment or directive line means nothing in a terminal. The
    # body of a `code-block` follows the directive indented, so it stays.
    text = re.sub(
        r"^[ \t]*\.\. .*(?:\n[ \t]*\n)?", "", text, flags=re.MULTILINE
    )
    text = re.sub(r"`([^`<]+?)(\s*)<([^>]+)>`_", r"\1\2(\3)", text)
    text = re.sub(r":\w+:`([^`]+)`", r"\1", text)
    text = text.replace("``", "")
    return re.sub(r"`([^`]+)`_?", r"\1", text)


if __name__ == "__main__":
    app.meta()
