"""The base class of the knowledge-distillation losses."""

from torch import Size, nn

from luxonis_train.attached_modules.losses.base_loss import BaseLoss
from luxonis_train.typing import Packet


class BaseDistillationLoss(BaseLoss, register=False):
    """Base class for the losses that compare a node with a teacher
    node.

    A distillation loss sits in the ``distillation`` list of a node, or
    the automatic recipe adds it. It runs only in training steps, and
    only while the model has a teacher. It gets the output packet of the
    matched teacher node next to the packet of its own node. Its
    `BaseLoss.forward` reads the teacher packet through parameters that
    start with ``teacher``, as `BaseAttachedModule.get_parameters`
    describes.

    A subclass with trainable parts, such as an adapter that maps the
    student channels to the teacher channels, creates them in `build`.
    The trainer registers the returned module, so the optimizer, DDP and
    the checkpoints cover it. The exported model never contains it.

    Attributes:
        teacher_node: The identifier of the teacher node that the loss
            reads. The trainer fills it in when it matches the nodes. A
            config sets it only to override the match.

    """

    def __init__(self, teacher_node: str | None = None, **kwargs):
        """Initialize the loss.

        Args:
            teacher_node: The identifier of the teacher node. ``None``
                lets the trainer match it.
            **kwargs: Keyword arguments forwarded to `BaseLoss`, such as
                ``final_loss_weight`` and ``node``.

        """
        super().__init__(**kwargs)
        self.teacher_node = teacher_node

    def build(
        self, student_shapes: Packet[Size], teacher_shapes: Packet[Size]
    ) -> nn.Module | None:
        """Create the trainable parts that depend on the output shapes.

        The trainer calls the method once, before training starts. The
        default creates nothing.

        Args:
            student_shapes: The output shapes of the node of the loss,
                with a batch dimension.
            teacher_shapes: The output shapes of the teacher node, with
                a batch dimension.

        Returns:
            The trainable module of the loss, or ``None`` when it has
            none.

        """
        _ = student_shapes, teacher_shapes
        return None
