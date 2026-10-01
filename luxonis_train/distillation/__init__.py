"""Knowledge distillation: train a model with a frozen teacher model.

Set ``model.teacher.weights`` to the checkpoint of a trained
luxonis-train model. When training starts,
`LuxonisLightningModule.attach_distillation` reads it:

- `recipe` matches the student nodes to the teacher nodes and chooses
  the distillation losses of every node.
- `teacher` builds a frozen copy of the teacher nodes that the losses
  read.
- `controller` adds the losses to the nodes and runs the teacher in
  every training step.

The losses themselves are in
`luxonis_train.attached_modules.losses.distillation`.

"""
