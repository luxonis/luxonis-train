"""The base class of the knowledge-distillation losses."""

from collections.abc import Mapping
from dataclasses import dataclass
from inspect import Parameter

from luxonis_ml.typing import Params
from torch import Size, Tensor, nn
from typing_extensions import override

from luxonis_train.attached_modules.losses.base_loss import BaseLoss
from luxonis_train.typing import Labels, Packet


@dataclass(frozen=True)
class StudentContext:
    """What the recipe knows about the node of a distillation loss.

    Attributes:
        losses: The task losses of the node, keyed by identifier.
        levels_read: The indices of the ``features`` levels that the
            nodes fed by the node read. ``None`` when no node reads them.

    """

    losses: Mapping[str, BaseLoss]
    levels_read: list[int] | None


class BaseDistillationLoss(BaseLoss, register=False):
    """Base class for the losses that compare a node with a teacher
    node.

    A distillation loss sits in the ``distillation`` list of a node, or
    the automatic recipe adds it. It runs only in training steps, and
    only while the model has a teacher.

    `BaseLoss.forward` reads the student packet by the rules of
    `BaseAttachedModule.get_parameters`, and the packet of the matched
    teacher node through the parameters that start with ``teacher``:
    ``teacher_features`` selects ``features``, and ``teacher`` selects
    the ``main_output`` of `BaseAttachedModule.task`.

    The trainer prepares a loss in two steps before training starts:

    - `derive_params` gives the constructor arguments that follow from
      the student node. The ``params`` of a config entry override them.
    - `setup` checks the teacher outputs and creates the trainable parts
      of the loss, such as an adapter from the student channels to the
      teacher channels. The trainer registers the returned module, so
      the optimizer, DDP and the checkpoints cover it. The exported
      model never contains it.

    """

    @classmethod
    def derive_params(cls, student: StudentContext) -> Params:
        """Derive constructor arguments from the student node.

        The default derives none.

        Args:
            student: The student node of the loss.

        Returns:
            The arguments, keyed by parameter name.

        """
        _ = student
        return {}

    def setup(
        self, student_shapes: Packet[Size], teacher_shapes: Packet[Size]
    ) -> nn.Module | None:
        """Check the teacher outputs and create the trainable parts.

        A subclass that overrides the method calls it first.

        Args:
            student_shapes: The output shapes of the node of the loss,
                with a batch dimension.
            teacher_shapes: The output shapes of the teacher node, with
                a batch dimension.

        Returns:
            The trainable module of the loss, or ``None`` when it has
            none.

        Raises:
            ValueError: When a required ``teacher`` parameter of
                `BaseLoss.forward` names an output that the teacher node does not
                give.

        """
        _ = student_shapes
        for name, parameter in self._signature.items():
            key = self._teacher_key(name)
            if (
                key is None
                or key in teacher_shapes
                or parameter.default is not Parameter.empty
                or self._argument_is_optional(parameter)
            ):
                continue
            raise ValueError(
                f"'{self.name}' reads the teacher output '{key}', but the "
                f"teacher node gives only {list(teacher_shapes)}."
            )
        return None

    @override
    def run(
        self,
        inputs: Packet[Tensor],
        labels: Labels,
        teacher: Packet[Tensor] | None = None,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Run the loss on the outputs of the node and its teacher node.

        Args:
            inputs: The output packet of the node.
            labels: The labels of the batch.
            teacher: The output packet of the teacher node. ``None``
                leaves every ``teacher`` parameter without a value.

        Returns:
            The result of `BaseLoss.run`.

        Example:
            >>> import torch
            >>> from torch import Tensor
            >>> class Loss(BaseDistillationLoss, register=False):
            ...     def forward(
            ...         self, features: Tensor, teacher_features: Tensor
            ...     ) -> Tensor:
            ...         return (features - teacher_features).abs().mean()
            >>> student = {"features": torch.zeros(2)}
            >>> teacher = {"features": torch.ones(2)}
            >>> Loss().run(student, {}, teacher).item()
            1.0

        """
        return super().run({**inputs, **self._teacher_inputs(teacher)}, labels)

    def _teacher_key(self, name: str) -> str | None:
        """Map a ``teacher`` parameter to its key in the teacher
        packet.
        """
        if name == "teacher":
            return self.task.main_output
        if name.startswith("teacher_"):
            return name.removeprefix("teacher_")
        return None

    def _teacher_inputs(
        self, teacher: Packet[Tensor] | None
    ) -> Packet[Tensor]:
        """Key the teacher packet by the ``teacher`` parameter names."""
        teacher = teacher or {}
        inputs = {f"teacher_{key}": value for key, value in teacher.items()}
        if "teacher" in self._signature:
            main_output = self.task.main_output
            if main_output in teacher:
                inputs["teacher"] = teacher[main_output]
        return inputs
