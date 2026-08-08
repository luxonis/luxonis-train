from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch import Size, Tensor, nn

from luxonis_train.nodes import BaseNode
from luxonis_train.nodes.blocks import GeneralReparametrizableBlock
from luxonis_train.tasks import Tasks
from luxonis_train.typing import AttachIndexType, Packet
from luxonis_train.utils import DatasetMetadata, IncompatibleError


class TensorNode(BaseNode, register=False):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + 1


class ListTensorNode(BaseNode, register=False):
    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        return [x + 1 for x in inputs]


class PacketNode(BaseNode, register=False):
    def forward(self, packet: Packet[Tensor]) -> Packet[Tensor]:
        return packet


class PacketListNode(BaseNode, register=False):
    def forward(self, packets: list[Packet[Tensor]]) -> Packet[Tensor]:
        return packets[0]


class InvalidPacketListNode(BaseNode, register=False):
    def forward(self, packets: list[Packet[Tensor]], inputs: Tensor) -> Tensor:
        return packets[0]["features"][0] + inputs


class TwoPacketNode(BaseNode, register=False):
    def forward(
        self, first: Packet[Tensor], second: Packet[Tensor]
    ) -> Packet[Tensor]:
        return first | second


class XYZNode(BaseNode, register=False):
    def forward(self, x: Tensor, y: Tensor, z: Tensor) -> Tensor:
        return x + y + z


class NamedNode(BaseNode, register=False):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return left + right


class OddNameNode(BaseNode, register=False):
    def forward(self, unusual: Tensor) -> Tensor:
        return unusual


class UnsupportedInputNode(BaseNode, register=False):
    def forward(self, inputs: int) -> Tensor:
        return torch.tensor(inputs)


class InvalidOutputNode(BaseNode, register=False):
    def forward(self, inputs: Tensor) -> int:
        return inputs.numel()


class TaskNode(TensorNode, register=False):
    task = Tasks.CLASSIFICATION


class CheckpointNode(BaseNode, register=False):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.linear = nn.Linear(2, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.linear(inputs)


class WeightedNode(TensorNode, register=False):
    def __init__(self, url: str, **kwargs):
        self.url = url
        super().__init__(**kwargs)

    def get_weights_url(self) -> str:
        return self.url


class ReparamNode(BaseNode, register=False):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.block = GeneralReparametrizableBlock(8, 8)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class InitializationNode(TensorNode, register=False):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_norm = nn.BatchNorm2d(2)
        self.activations = nn.ModuleList(
            [
                nn.Hardswish(inplace=False),
                nn.LeakyReLU(inplace=False),
                nn.ReLU(inplace=False),
                nn.ReLU6(inplace=False),
                nn.SiLU(inplace=False),
            ]
        )


def test_constructor_properties_and_post_init():
    node = TensorNode(
        attach_index=0,
        remove_on_export=True,
        export_output_names=["output"],
        original_in_shape=Size((3, 16, 16)),
        n_classes=2,
        n_keypoints=3,
    )
    assert node.attach_index == 0
    assert node.name == "TensorNode"
    assert node.n_classes == 2
    assert node.n_keypoints == 3
    assert node.remove_on_export
    assert node.export_output_names == ["output"]
    assert node.original_in_shape == Size((3, 16, 16))

    node._variant = None
    with pytest.raises(RuntimeError, match="Variant was not set"):
        _ = node.variant
    node._variant = "tiny"
    assert node.variant == "tiny"

    node.load_checkpoint = Mock()
    node._weights = "download"
    node.__post_init__()
    node.load_checkpoint.assert_called_once_with()
    node.load_checkpoint.reset_mock()
    node._weights = "https://example.com/weights.ckpt"
    node.__post_init__()
    node.load_checkpoint.assert_called_once_with(
        ckpt="https://example.com/weights.ckpt"
    )
    node.initialize_weights = Mock()
    node._weights = "none"
    node.__post_init__()
    node.initialize_weights.assert_called_once_with(method="none")


def test_yolo_weight_initialization():
    node = InitializationNode()
    node.initialize_weights("yolo")

    assert node.batch_norm.eps == 1e-3
    assert node.batch_norm.momentum == 3e-2
    assert all(activation.inplace for activation in node.activations)


def test_metadata_and_required_properties():
    metadata = DatasetMetadata(
        classes={"task": {"second": 1, "first": 0}},
        n_keypoints={"task": 4},
    )
    node = TensorNode(dataset_metadata=metadata, task_name="task")

    assert node.dataset_metadata is metadata
    assert node.n_classes == 2
    assert node.n_keypoints == 4
    assert dict(node.classes) == {"second": 1, "first": 0}
    assert node.class_names == ["first", "second"]

    missing = TensorNode()
    with pytest.raises(RuntimeError, match="input_shapes"):
        _ = missing.input_shapes
    with pytest.raises(RuntimeError, match="original_in_shape"):
        _ = missing.original_in_shape
    with pytest.raises(RuntimeError, match="dataset_metadata"):
        _ = missing.dataset_metadata


def test_in_sizes_from_input_shapes():
    size1 = Size((8, 16, 16))
    size2 = Size((16, 8, 8))

    with pytest.raises(RuntimeError, match="multiple preceding"):
        _ = TensorNode(
            input_shapes=[{"features": [size1]}, {"features": [size2]}]
        ).in_sizes

    node = TensorNode(input_shapes=[{"custom": [size1]}])
    assert node.in_sizes == size1

    with pytest.raises(RuntimeError, match="Unable to determine"):
        _ = NamedNode(
            input_shapes=[{"other": [size1], "another": [size1]}]
        ).in_sizes

    one_match = NamedNode(
        input_shapes=[{"left": [size1], "other": [size2]}],
        attach_index=-1,
    )
    assert one_match.in_sizes == size1

    equal = NamedNode(
        input_shapes=[{"left": [size1], "right": [size1]}],
        attach_index=-1,
    )
    assert equal.in_sizes == size1

    different = NamedNode(
        input_shapes=[{"left": [size1], "right": [size2]}],
        attach_index=-1,
    )
    with pytest.raises(RuntimeError, match="different shapes"):
        _ = different.in_sizes

    features = ListTensorNode(input_shapes=[{"features": [size1, size2]}])
    assert features.in_sizes == [size1, size2]
    assert features.in_channels == [8, 16]
    assert features.in_height == [16, 8]
    assert features.in_width == [16, 8]

    direct = TensorNode(in_sizes=size1)
    assert direct.in_channels == 8
    assert direct.in_height == 16
    assert direct.in_width == 16


@pytest.mark.parametrize(
    ("attach_index", "expected"),
    [
        ("all", [1, 2, 3, 4, 5]),
        (0, 1),
        (-1, 5),
        ((0, 2), [1, 2]),
        ((-1, -3), [5, 4]),
        ((-3, -1), [4, 5]),
        ((-2, 3), []),
        ((1, -1), [2, 3, 4]),
        ((4, 2), [5, 4]),
        ((1, 4), [2, 3, 4]),
        ((0, 4, 2), [1, 3]),
    ],
)
def test_get_attached(
    attach_index: AttachIndexType, expected: list[int] | int
):
    node = TensorNode(attach_index=attach_index)
    assert node.get_attached([1, 2, 3, 4, 5]) == expected  # type: ignore[arg-type]


def test_get_attached_errors_and_scalar():
    node = TensorNode()
    assert node.get_attached(torch.tensor(1)) == 1

    node.attach_index = "all"
    with pytest.raises(ValueError, match="input is not a list"):
        node.get_attached(torch.tensor(1))
    node.attach_index = 10
    with pytest.raises(ValueError, match="out of range"):
        node.get_attached([1, 2, 3])  # type: ignore[arg-type]
    node.attach_index = None
    with pytest.raises(RuntimeError, match="Attach index not defined"):
        node.get_attached([1, 2, 3])  # type: ignore[arg-type]


def test_weights_url_expansion():
    assert TensorNode()._get_weights_url() is None

    node = WeightedNode("{github}/{variant}/weights.ckpt")
    node._variant = None
    with pytest.raises(ValueError, match="placeholder"):
        node._get_weights_url()

    node._variant = "tiny"
    assert node._get_weights_url() == (
        "https://github.com/luxonis/luxonis-train/releases/download/"
        "v0.3.10-beta//tiny/weights.ckpt"
    )
    tagged = WeightedNode("{github:v1.2.3-rc1}/weights.ckpt")
    assert tagged._get_weights_url() == (
        "https://github.com/luxonis/luxonis-train/releases/download/"
        "v1.2.3-rc1//weights.ckpt"
    )
    assert WeightedNode("local.ckpt")._get_weights_url() == "local.ckpt"


def test_load_checkpoint(monkeypatch: pytest.MonkeyPatch):
    node = CheckpointNode()
    with pytest.raises(RuntimeError, match="dictionary is empty"):
        node.load_checkpoint({})
    with pytest.raises(ValueError, match="does not implement"):
        node.load_checkpoint()

    state_dict = node.state_dict()
    node.load_checkpoint(state_dict)

    load_state_dict = Mock()
    monkeypatch.setattr(node, "load_state_dict", load_state_dict)
    monkeypatch.setattr(
        "luxonis_train.nodes.base_node.safe_download", lambda _: None
    )
    node.load_checkpoint("missing.ckpt")
    load_state_dict.assert_not_called()

    monkeypatch.setattr(
        "luxonis_train.nodes.base_node.safe_download",
        lambda _: Path("downloaded.ckpt"),
    )
    loaded_state = {"linear.bias": torch.ones(2)}
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_, **__: {"state_dict": loaded_state},
    )
    node.load_checkpoint("remote.ckpt", strict=False)
    load_state_dict.assert_called_once_with(loaded_state, strict=False)


def test_export_mode_reparametrizes_modules():
    node = ReparamNode(remove_on_export=True).eval()
    block = node.block
    assert block.fused_branch is None

    node.export = True
    assert node.export
    assert block.fused_branch is not None
    node.export = False
    assert not node.export
    assert block.fused_branch is None


def test_run_packet_and_packet_list_inputs():
    tensor = torch.randn(2, 3)
    packets: list[Packet[Tensor]] = [{"features": [tensor]}]
    assert PacketListNode().run(packets) is packets[0]
    assert PacketNode().run(packets) is packets[0]

    with pytest.raises(RuntimeError, match="only parameter"):
        InvalidPacketListNode().run(packets)
    with pytest.raises(RuntimeError, match="at least 2 inputs"):
        TwoPacketNode().run(packets)


def test_run_feature_inputs_and_outputs():
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    output = TensorNode().run([{"features": x}])["features"]
    assert isinstance(output, Tensor)
    assert torch.equal(output, x + 1)
    list_output = ListTensorNode().run([{"features": [x, y]}])
    output_list = list_output["features"]
    assert isinstance(output_list, list)
    assert len(output_list) == 2
    with pytest.raises(RuntimeError, match="single tensor"):
        ListTensorNode().run([{"features": x}])
    with pytest.raises(RuntimeError, match="was not found"):
        TensorNode().run([{"other": x}])

    xyz = XYZNode(attach_index=-1).run(
        [{"features": x}, {"features": x}, {"features": x}]
    )
    xyz_features = xyz["features"]
    assert isinstance(xyz_features, Tensor)
    assert torch.equal(xyz_features, x * 3)

    task_output = TaskNode().run([{"features": x}])
    assert set(task_output) == {"classification"}
    with pytest.raises(ValueError, match="Invalid output type"):
        InvalidOutputNode().run([{"features": x}])


def test_run_named_and_unusual_inputs():
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    node = NamedNode(attach_index=-1)
    output = node.run([{"left": x}, {"right": y}])
    output_features = output["features"]
    assert isinstance(output_features, Tensor)
    assert torch.equal(output_features, x + y)

    with pytest.raises(RuntimeError, match="type"):
        node.run([{"left": [x]}, {"right": y}])
    with pytest.raises(RuntimeError, match="multiple input packets"):
        node.run([{"left": x}, {"left": x, "right": y}])

    unusual = OddNameNode(attach_index=-1).run([{"custom": [x]}])
    assert unusual["features"] is x
    with pytest.raises(TypeError, match="unsupported type"):
        UnsupportedInputNode().run([{"features": x}])


def test_type_override_validation():
    class IncompatibleNode(BaseNode, register=False):
        in_channels: int
        attach_index = "all"

        def forward(self, inputs: list[Tensor]) -> list[Tensor]:
            return inputs

    with pytest.raises(IncompatibleError):
        IncompatibleNode(in_sizes=[Size((3, 4, 4)), Size((3, 2, 2))])
