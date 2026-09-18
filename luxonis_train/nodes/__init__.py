"""The building blocks of a model graph.

A node is one computational unit of the model. An entry in the
``model.nodes`` section of the config names a node class. The
``inputs`` field of the entry lists the nodes that feed the node. The
``input_sources`` field lists the loader outputs that feed it. Every
node inherits `BaseNode`.

- `luxonis_train.nodes.backbones` holds the backbones. A backbone
  turns an image into feature maps.
- `luxonis_train.nodes.necks` holds the necks. A neck fuses the
  feature maps of a backbone.
- `luxonis_train.nodes.heads` holds the heads. A head turns features
  into predictions for a task.
- `luxonis_train.nodes.blocks` holds the layers that several nodes
  share.

A head carries a task. The task decides which losses, metrics, and
visualizers can attach to the head. The docstring of each head lists
them.

"""

from .backbones import *
from .base_node import *
from .heads import *
from .necks import *
