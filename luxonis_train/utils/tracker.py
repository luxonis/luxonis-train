from typing import TYPE_CHECKING, Any

from lightning.pytorch.loggers.logger import Logger
from luxonis_ml.tracker import LuxonisTracker
from typing_extensions import override

if TYPE_CHECKING:
    from luxonis_train.config.config import TrackerConfig


class LuxonisTrackerPL(LuxonisTracker, Logger):
    """A C{LuxonisTracker} that PyTorch Lightning can use as its logger.

    The run of the main tracker of a L{LuxonisModel} stays open after
    training and testing, so that a later export or archive still
    uploads to it. It closes when the process exits, or at an explicit
    L{close}. A failed training or test marks the run as failed through
    L{mark_failed}, and the run then ends as failed.
    """

    def __init__(self, *, _auto_finalize: bool = True, **kwargs: Any):
        """Create the tracker.

        @type _auto_finalize: bool
        @param _auto_finalize: If C{True}, the run closes when Lightning
            finalizes the logger at the end of C{fit}, as for a tuning
            trial. If C{False}, the run closes only at L{close}, or when
            the process exits.
        @type kwargs: Any
        @param kwargs: The keyword arguments of C{LuxonisTracker}.
        """
        LuxonisTracker.__init__(self, **kwargs)
        Logger.__init__(self)
        self._failed = False
        if _auto_finalize:
            self.finalize = self.close

    def mark_failed(self) -> None:
        """Mark the run as failed.

        The run stays open. It ends as failed whenever it closes.
        """
        self._failed = True

    @override
    def close(self, status: str = "success") -> None:
        """End the run, as failed if L{mark_failed} ran before.

        @type status: str
        @param status: The status of the run. See
            C{LuxonisTracker.close}.
        """
        super().close("failed" if self._failed else status)


def get_tracker_init_params(cfg_tracker: "TrackerConfig") -> dict[str, Any]:
    """Turn the tracker config into the arguments of the tracker.

    @type cfg_tracker: L{TrackerConfig}
    @param cfg_tracker: The tracker section of the config.
    @rtype: dict[str, Any]
    @return: The keyword arguments of L{LuxonisTrackerPL}.
    """
    tracker_params = cfg_tracker.model_dump()
    tracker_params.update(tracker_params.pop("plugins"))
    tracker_params["save_directory"] = cfg_tracker.save_directory
    return tracker_params
