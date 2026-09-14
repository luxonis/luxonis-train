"""The filter for the checkpoint keys of the attached modules.

A loss, a metric, or a visualizer stores its node in ``_node``. A state
dict that includes such a module holds the parameters and the buffers of
the node again, under the key of the module.
`CHECKPOINT_FILTERED_STATE_DICT_PATTERN` matches these keys, and
`filter_checkpoint_state_dict` drops them.

A key matches the pattern when it starts with ``nodes.<node>.losses.``,
``nodes.<node>.metrics.``, or ``nodes.<node>.visualizers.``, and holds
``_node.`` after that prefix. ``<node>`` is a name without a dot. For
example, ``nodes.head.losses.loss._node.weight`` matches.

`LuxonisLightningModule` drops these keys when it saves a checkpoint. It
also ignores them when a run resumes with strict weight loading.
`EMACallback.state_dict` drops them from the average that it returns.
`EMACallback.on_fit_start` ignores them when it restores an average from
a checkpoint.

"""

import re
from collections.abc import Mapping

from torch import Tensor

CHECKPOINT_FILTERED_STATE_DICT_PATTERN = re.compile(
    r"^nodes\.[^.]+\.(metrics|visualizers|losses)\..*_node\..*"
)


def filter_checkpoint_state_dict(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Drop the keys that the attached modules hold for their node.

    The function returns a new dictionary without the keys that match
    `CHECKPOINT_FILTERED_STATE_DICT_PATTERN`. It does not change
    ``state_dict``.

    Args:
        state_dict (``Mapping[str, Tensor]``): The state dict of the model,
            keyed by the names of the parameters and the buffers.

    Returns:
        ``dict[str, Tensor]``: The entries of ``state_dict`` with a key
        that does not match the pattern, in the same order.

    Example:
        >>> import torch
        >>> state_dict = {
        ...     "nodes.head.module.weight": torch.ones(1),
        ...     "nodes.head.losses.loss._node.weight": torch.ones(1),
        ... }
        >>> list(filter_checkpoint_state_dict(state_dict))
        ['nodes.head.module.weight']

    """
    return {
        key: value
        for key, value in state_dict.items()
        if not CHECKPOINT_FILTERED_STATE_DICT_PATTERN.match(key)
    }
