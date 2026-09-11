"""Filters the checkpoint keys that belong to a loss, a metric, or a
visualizer, which lets a checkpoint load into a changed config.
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
    return {
        key: value
        for key, value in state_dict.items()
        if not CHECKPOINT_FILTERED_STATE_DICT_PATTERN.match(key)
    }
