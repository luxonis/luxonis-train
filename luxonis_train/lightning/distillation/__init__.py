"""Knowledge distillation: train a model with a frozen teacher model.

Set ``model.teacher.weights`` to the checkpoint of a trained
luxonis-train model. When ``fit`` starts,
`LuxonisLightningModule.setup` builds a `Distiller`:

- `recipe` matches the student nodes to the teacher nodes and chooses
  the distillation losses of every node.
- `teacher` builds a frozen copy of the teacher nodes that the losses
  read.
- `distiller` holds the teacher and the losses, and runs them in every
  training step.

The losses themselves are in
`luxonis_train.attached_modules.losses.distillation`.

"""

from .distiller import Distiller, ReleaseTeacherCallback

__all__ = ["Distiller", "ReleaseTeacherCallback"]
