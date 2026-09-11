"""The building blocks of a model graph.

A node is one computational unit. Connect the nodes through the
``inputs`` field of the config. Two nodes connect when the shapes
agree.

- `luxonis_train.nodes.backbones` extract features from an image
- `luxonis_train.nodes.necks` fuse features across scales
- `luxonis_train.nodes.heads` turn features into predictions for a task

A head carries a task, which decides the losses, the metrics, and the
visualizers that can attach to it. Every node docstring lists them.

"""

from .backbones import *
from .base_node import *
from .heads import *
from .necks import *
