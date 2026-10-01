"""The frozen teacher model of knowledge distillation."""

from collections.abc import Collection, Mapping
from typing import Any

import torch
from loguru import logger
from semver import Version
from torch import Size, Tensor, nn
from typing_extensions import Self, override

from luxonis_train.config import Config, NodeConfig, TeacherConfig
from luxonis_train.config.config import ModelConfig, PreprocessingConfig
from luxonis_train.lightning.utils import Nodes, node_inputs, node_state_dict
from luxonis_train.typing import Packet
from luxonis_train.upgrade import upgrade_config
from luxonis_train.utils import DatasetMetadata, safe_download

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class InputAdapter(nn.Module):
    """Convert a student input batch to the preprocessing of the
    teacher.

    The loader normalizes the images for the student. When the teacher
    was trained with another normalization or another channel order, the
    adapter undoes the student normalization, swaps RGB and BGR, and
    applies the teacher normalization. Without differences, it returns
    the inputs unchanged.

    """

    raw_scale: Tensor
    raw_shift: Tensor
    scale: Tensor
    shift: Tensor

    def __init__(
        self,
        image_source: str,
        student: PreprocessingConfig,
        teacher: PreprocessingConfig,
    ):
        """Derive the conversion from the two preprocessing configs.

        Args:
            image_source: The name of the image input.
            student: The preprocessing of the student run.
            teacher: The preprocessing the teacher was trained with.

        Raises:
            ValueError: When only one of the two models trains on
                ``GRAY`` images.

        """
        super().__init__()
        if "GRAY" in {student.color_space, teacher.color_space} and (
            student.color_space != teacher.color_space
        ):
            raise ValueError(
                f"The teacher trains on {teacher.color_space} images and "
                f"the student on {student.color_space} images. A teacher "
                "and a student must both use GRAY, or neither."
            )
        self.image_source = image_source
        self.flip_channels = student.color_space != teacher.color_space
        self.identity = not self.flip_channels and _normalization(
            student
        ) == _normalization(teacher)
        to_raw, to_teacher = _raw_affine(student), _teacher_affine(teacher)
        self.register_buffer("raw_scale", to_raw[0], persistent=False)
        self.register_buffer("raw_shift", to_raw[1], persistent=False)
        self.register_buffer("scale", to_teacher[0], persistent=False)
        self.register_buffer("shift", to_teacher[1], persistent=False)

    @override
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert the image input; the other inputs pass unchanged.

        Args:
            inputs: The loader inputs of the student.

        Returns:
            The inputs for the teacher.

        """
        if self.identity:
            return inputs
        image = inputs[self.image_source] * self.raw_scale + self.raw_shift
        if self.flip_channels:
            image = image.flip(1)
        return {
            **inputs,
            self.image_source: image * self.scale + self.shift,
        }


class Teacher(nn.Module):
    """A frozen, pruned copy of a trained model.

    The teacher holds only the nodes that the distillation losses read
    and the nodes that feed them. Its parameters never get a gradient,
    and it stays in evaluation mode: `train` ignores its argument, so
    the batch norm statistics and the dropout of the teacher never
    change.

    `LuxonisLightningModule` does not register the teacher as a
    submodule. The teacher is therefore not in the state dict, the
    checkpoints, the EMA copy, the optimizer, or the DDP wrapper.

    Attributes:
        nodes: The nodes of the teacher.
        outputs: The identifiers of the nodes whose output packets
            `forward` returns.
        input_adapter: The conversion of the student inputs.

    """

    def __init__(
        self,
        nodes: Nodes,
        outputs: Collection[str],
        input_adapter: InputAdapter,
    ):
        """Freeze the nodes and put them in evaluation mode.

        Args:
            nodes: The nodes of the teacher, with weights.
            outputs: The identifiers of the nodes to return.
            input_adapter: The conversion of the student inputs.

        """
        super().__init__()
        self.nodes = nodes
        self.outputs = frozenset(outputs)
        self.input_adapter = input_adapter
        self.requires_grad_(False)
        self.train(False)

    @override
    def train(self, mode: bool = True) -> Self:
        """Keep the teacher in evaluation mode.

        Args:
            mode: Ignored.

        Returns:
            The teacher, in evaluation mode.

        """
        _ = mode
        return super().train(False)

    @override
    @torch.no_grad()
    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Packet[Tensor]]:
        """Run the nodes and return the packets of the output nodes.

        The method runs under ``torch.no_grad`` rather than
        ``torch.inference_mode``, because inference tensors cannot be
        saved for the backward pass of a loss.

        Args:
            inputs: The loader inputs of the student batch.

        Returns:
            The output packet of each node in `outputs`.

        """
        inputs = self.input_adapter(inputs)
        computed: dict[str, Packet[Tensor]] = {}
        for node_name, node, _, _ in self.nodes.traverse():
            computed[node_name] = node.module.run(
                node_inputs(node.inputs, computed, inputs)
            )
        return {name: computed[name] for name in self.outputs}


def load_teacher_checkpoint(cfg: TeacherConfig) -> dict[str, Any]:
    """Download and read the checkpoint of a teacher.

    Args:
        cfg: The teacher config.

    Returns:
        The checkpoint. Its ``config`` is migrated to the schema of the
        installed release.

    Raises:
        RuntimeError: When the download fails.
        ValueError: When the checkpoint has no ``config`` or no
            ``state_dict``. Only a luxonis-train checkpoint can be a
            teacher.

    """
    path = safe_download(cfg.weights)
    if path is None:
        raise RuntimeError(f"Failed to download the teacher '{cfg.weights}'.")
    ckpt = torch.load(path, map_location="cpu")  # nosemgrep
    missing = {"config", "state_dict"} - set(ckpt)
    if missing:
        raise ValueError(
            f"The teacher checkpoint '{cfg.weights}' has no "
            f"{' and no '.join(sorted(missing))}. A teacher must be a "
            "checkpoint that luxonis-train saved."
        )
    ckpt["config"] = upgrade_config(ckpt["config"])
    return ckpt


def teacher_node_configs(ckpt: Mapping[str, Any]) -> dict[str, NodeConfig]:
    """Read the node configs of a teacher checkpoint.

    The configs keep only what the teacher graph needs. They drop the
    attached modules, the freezing, the fine-tuning rules, and the
    metadata label overrides. The last ones would rename the labels of
    the student, because the task objects are shared. A node never
    downloads pretrained weights, because the checkpoint replaces them.

    Args:
        ckpt: A checkpoint from `load_teacher_checkpoint`.

    Returns:
        The node configs, keyed by node identifier, in graph order.

    """
    configs = [
        NodeConfig.model_validate(_clean_node(node))
        for node in ckpt["config"]["model"]["nodes"]
    ]
    return {cfg.identifier: cfg for cfg in configs}


def build_teacher(
    ckpt: Mapping[str, Any],
    node_configs: Mapping[str, NodeConfig],
    outputs: Collection[str],
    student_cfg: Config,
    input_shapes: dict[str, Size],
    *,
    strict: bool,
) -> Teacher:
    """Build the teacher nodes that ``outputs`` need and load weights.

    Args:
        ckpt: A checkpoint from `load_teacher_checkpoint`.
        node_configs: The node configs of `teacher_node_configs`.
        outputs: The identifiers of the teacher nodes that the
            distillation losses read.
        student_cfg: The config of the student run. The teacher reuses
            its loader and trainer sections.
        input_shapes: The shapes of the loader inputs of the student,
            without the batch dimension.
        strict: Whether a node must load without missing or unexpected
            keys.

    Returns:
        The frozen teacher.

    Raises:
        RuntimeError: When a node fails to load with ``strict``, or
            when the checkpoint has no weights for a node.

    """
    needed = _ancestors(node_configs, outputs)
    model = ModelConfig.model_construct(
        name="teacher",
        nodes=[cfg for name, cfg in node_configs.items() if name in needed],
        outputs=sorted(outputs),
    )
    cfg = student_cfg.model_copy(update={"model": model})
    metadata = DatasetMetadata(**ckpt.get("dataset_metadata", {}))
    nodes = Nodes(cfg, metadata, input_shapes)
    _load_weights(nodes, ckpt, strict=strict)
    teacher_preprocessing = PreprocessingConfig.model_validate(
        ckpt["config"].get("trainer", {}).get("preprocessing", {})
    )
    adapter = InputAdapter(
        student_cfg.loader.image_source,
        student_cfg.trainer.preprocessing,
        teacher_preprocessing,
    )
    return Teacher(nodes, outputs, adapter)


def _clean_node(node: dict[str, Any]) -> dict[str, Any]:
    params = dict(node.get("params") or {})
    if "weights" in params:
        params["weights"] = "none"
    cleaned = {
        **node,
        "params": params,
        "losses": [],
        "metrics": [],
        "visualizers": [],
        "finetuning": [],
        "distillation": "off",
        "metadata_task_override": None,
    }
    cleaned.pop("freezing", None)
    return cleaned


def _ancestors(
    node_configs: Mapping[str, NodeConfig], outputs: Collection[str]
) -> set[str]:
    needed: set[str] = set()
    stack = list(outputs)
    while stack:
        name = stack.pop()
        if name not in needed:
            needed.add(name)
            stack.extend(node_configs[name].inputs)
    return needed


def _load_weights(
    nodes: Nodes, ckpt: Mapping[str, Any], *, strict: bool
) -> None:
    version = Version.parse(ckpt.get("version", "0.3.0"))
    for name, node in nodes.items():
        weights = node_state_dict(ckpt["state_dict"], name, version)
        if not weights and node.module.state_dict():
            raise RuntimeError(
                f"The teacher checkpoint has no weights for node '{name}'."
            )
        try:
            result = node.module.load_state_dict(weights, strict=strict)
        except RuntimeError as error:
            raise RuntimeError(
                f"Teacher node '{name}' does not match its weights in the "
                "checkpoint. Set `model.teacher.strict` to false to load "
                "it anyway."
            ) from error
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                f"Teacher node '{name}' loaded with "
                f"{len(result.missing_keys)} missing and "
                f"{len(result.unexpected_keys)} unexpected keys."
            )


def _normalization(cfg: PreprocessingConfig) -> tuple[Any, Any] | None:
    if not cfg.normalize.active:
        return None
    params = cfg.normalize.params
    return params.get("mean", _IMAGENET_MEAN), params.get("std", _IMAGENET_STD)


def _raw_affine(cfg: PreprocessingConfig) -> tuple[Tensor, Tensor]:
    # Normalized input -> pixel values in [0, 255].
    normalization = _normalization(cfg)
    if normalization is None:
        return torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
    mean, std = (_channels(values) for values in normalization)
    return 255 * std, 255 * mean


def _teacher_affine(cfg: PreprocessingConfig) -> tuple[Tensor, Tensor]:
    # Pixel values in [0, 255] -> normalized teacher input.
    normalization = _normalization(cfg)
    if normalization is None:
        return torch.ones(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)
    mean, std = (_channels(values) for values in normalization)
    return 1 / (255 * std), -mean / std


def _channels(values: Any) -> Tensor:
    return torch.tensor(values, dtype=torch.float32).view(1, -1, 1, 1)
