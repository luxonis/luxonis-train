"""The Lightning module that runs the graph.

`LuxonisLightningModule` builds the nodes from a config, runs them in
topological order, and computes the losses, the metrics, and the
visualizations. `LuxonisOutput` holds the result of one forward pass.

"""

from .luxonis_lightning import LuxonisLightningModule
from .luxonis_output import LuxonisOutput

__all__ = ["LuxonisLightningModule", "LuxonisOutput"]
