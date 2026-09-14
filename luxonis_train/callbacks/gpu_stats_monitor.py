"""Logs the GPU statistics of ``nvidia-smi`` while a model trains.

Copyright The PyTorch Lightning team.

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

import os
import shutil
import subprocess
import time
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.accelerators.cuda import CUDAAccelerator
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.utilities.parsing import AttributeDict
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lightning_fabric.utilities.exceptions import MisconfigurationException

from luxonis_train.registry import CALLBACKS


@CALLBACKS.register()
class GPUStatsMonitor(pl.Callback):
    """Callback that logs the GPU statistics of ``nvidia-smi``.

    The callback queries ``nvidia-smi --query-gpu`` for each GPU of the
    trainer at the start and at the end of a train batch. It logs the
    values with the logger of the trainer, at ``trainer.global_step``.
    It logs only on the steps where the trainer updates its logs.
    Lightning selects these steps with ``log_every_n_steps`` of the
    trainer. The batch hooks run on rank 0 only.

    Each flag of `GPUStatsMonitor.__init__` turns on a set of queries:

    - ``gpu_utilization``: ``utilization.gpu``, in percent. It is the
      part of the last sample period in which one or more kernels ran
      on the GPU. The callback logs it at the start and at the end of a
      batch.
    - ``memory_utilization``: ``memory.used`` and ``memory.free``, in
      MiB, and ``utilization.memory``, in percent. The last value is the
      part of the last sample period in which the GPU read or wrote its
      memory. The callback logs them at the start and at the end of a
      batch.
    - ``fan_speed``: ``fan.speed``, in percent of the maximum speed. It
      is the intended speed, not a measured speed. The callback logs it
      at the end of a batch only.
    - ``temperature``: ``temperature.gpu`` and ``temperature.memory``,
      in degrees Celsius. The callback logs them at the end of a batch
      only.

    A metric name has the form ``GPU_<device>/<query> - <unit>``, for
    example ``GPU_0/utilization.gpu - percent`` or
    ``GPU_0/memory.used - MB``. ``<device>`` is the logical device
    index of the trainer. ``<unit>`` is ``percent``, ``MB``, or
    ``°C``. The ``MB`` label marks a value in MiB. The callback logs
    ``0.0`` for a value that is not a number, such as ``[N/A]``.

    Two more flags log the time of the batches, in milliseconds:

    - ``intra_step_time``: ``batch_time/intra_step (ms)``, the time
      from the start to the end of a batch. The callback logs it at the
      end of a batch.
    - ``inter_step_time``: ``batch_time/inter_step (ms)``, the time
      from the end of the previous batch to the start of this batch.
      The callback logs it at the start of a batch. The first batch of
      an epoch has no value.

    Each time includes the ``nvidia-smi`` queries that run in its
    interval.

    The callback is in the ``CALLBACKS`` registry, so a config can add
    it:

    .. code-block:: yaml

        trainer:
          callbacks:
            - name: GPUStatsMonitor
              params:
                temperature: true
                intra_step_time: true

    """

    def __init__(
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False,
    ):
        """Initialize the callback with the statistics to log.

        The class docstring describes each statistic. The constructor
        checks only for ``nvidia-smi``. `GPUStatsMonitor.setup` checks
        for a logger and for CUDA.

        Args:
            memory_utilization (bool): Log ``memory.used``,
                ``memory.free``, and ``utilization.memory`` at the start
                and at the end of a batch.
            gpu_utilization (bool): Log ``utilization.gpu`` at the start
                and at the end of a batch.
            intra_step_time (bool): Log the time from the start to the
                end of a batch as ``batch_time/intra_step (ms)``.
            inter_step_time (bool): Log the time from the end of a batch
                to the start of the next batch as
                ``batch_time/inter_step (ms)``.
            fan_speed (bool): Log ``fan.speed`` at the end of a batch.
            temperature (bool): Log ``temperature.gpu`` and
                ``temperature.memory`` at the end of a batch.

        Raises:
            MisconfigurationException: When the ``nvidia-smi``
                executable is not on the ``PATH``. The message says
                that the NVIDIA driver is not installed.

        """
        super().__init__()

        if shutil.which("nvidia-smi") is None:
            raise MisconfigurationException(
                "Cannot use GPUStatsMonitor callback because NVIDIA driver is not installed."
            )

        self._log_stats = AttributeDict(
            {
                "memory_utilization": memory_utilization,
                "gpu_utilization": gpu_utilization,
                "intra_step_time": intra_step_time,
                "inter_step_time": inter_step_time,
                "fan_speed": fan_speed,
                "temperature": temperature,
            }
        )

        # The logical device IDs for selected devices
        self._device_ids: list[int] = []  # will be assigned later in setup()

        # The unmasked real GPU IDs
        self._gpu_ids: list[str] = []  # will be assigned later in setup()

    @staticmethod
    def is_available() -> bool:
        """Return whether this machine can run the callback.

        The constructor and `GPUStatsMonitor.setup` do not call this
        method. They do their own checks and raise an error instead.

        Returns:
            bool: ``True`` when the ``nvidia-smi`` executable is on the
            ``PATH`` and the Lightning ``CUDAAccelerator`` reports CUDA
            as available.

        """
        if shutil.which("nvidia-smi") is None:
            return False
        return CUDAAccelerator.is_available()

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str | None = None,
    ) -> None:
        """Check the trainer and find the GPUs to query.

        Lightning calls this hook at the start of every stage. The hook
        stores the sorted, unique device indices of ``trainer``. It maps
        each index to a physical GPU ID through the
        ``CUDA_VISIBLE_DEVICES`` environment variable. When the variable
        is not set, each GPU ID is equal to its index.

        Args:
            trainer (``pl.Trainer``): The trainer. It must have a
                logger.
            pl_module (``pl.LightningModule``): The model. Unused.
            stage (str | None): The stage that starts. Unused.

        Raises:
            MisconfigurationException: When ``trainer`` has no logger,
                or when CUDA is not available.

        """
        if not trainer.logger:
            raise MisconfigurationException(
                "Cannot use GPUStatsMonitor callback with Trainer that has no logger."
            )

        if not CUDAAccelerator.is_available():
            raise MisconfigurationException(
                "You are using GPUStatsMonitor but the CUDA Accelerator is not available."
            )

        # The logical device IDs for selected devices
        self._device_ids = sorted(set(trainer.device_ids))

        # The unmasked real GPU IDs
        self._gpu_ids = self._get_gpu_ids(self._device_ids)

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Clear the recorded batch times.

        Lightning calls this hook at the start of every train epoch, on
        every rank. Because of the reset, the first batch of an epoch
        logs no ``batch_time/inter_step (ms)`` value.

        Args:
            trainer (``pl.Trainer``): The trainer. Unused.
            pl_module (``pl.LightningModule``): The model. Unused.

        """
        self._snap_intra_step_time: float | None = None
        self._snap_inter_step_time: float | None = None

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log the GPU utilization and memory before a batch.

        Lightning calls this hook before every train batch. The hook
        runs on rank 0 only. When ``intra_step_time`` is on, the hook
        first records the start time of the batch. It stops there when
        the trainer does not update its logs on this step.

        Otherwise the hook queries ``nvidia-smi`` for the statistics of
        the enabled flags among ``gpu_utilization`` and
        ``memory_utilization``. When ``inter_step_time`` is on and the
        previous batch recorded its end time, the hook adds
        ``batch_time/inter_step (ms)``. It logs the values with
        ``trainer.logger.log_metrics`` at ``trainer.global_step``.

        Args:
            trainer (``pl.Trainer``): The trainer. Its logger receives
                the values.
            pl_module (``pl.LightningModule``): The model. Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch. Unused.

        """
        if self._log_stats.intra_step_time:
            self._snap_intra_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        gpu_stat_keys = self._get_gpu_stat_keys()
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(
            self._device_ids, gpu_stats, gpu_stat_keys
        )

        if self._log_stats.inter_step_time and self._snap_inter_step_time:
            # First log at beginning of second step
            logs["batch_time/inter_step (ms)"] = (
                time.time() - self._snap_inter_step_time
            ) * 1000

        assert trainer.logger is not None
        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log all enabled GPU statistics after a batch.

        Lightning calls this hook after every train batch. The hook
        runs on rank 0 only. When ``inter_step_time`` is on, the hook
        first records the end time of the batch. It stops there when
        the trainer does not update its logs on this step.

        Otherwise the hook queries ``nvidia-smi`` for the statistics of
        the enabled flags among ``gpu_utilization``,
        ``memory_utilization``, ``fan_speed``, and ``temperature``.
        When ``intra_step_time`` is on, the hook adds
        ``batch_time/intra_step (ms)``. It logs the values with
        ``trainer.logger.log_metrics`` at ``trainer.global_step``.

        Args:
            trainer (``pl.Trainer``): The trainer. Its logger receives
                the values.
            pl_module (``pl.LightningModule``): The model. Unused.
            outputs (``STEP_OUTPUT``): The output of the training step.
                Unused.
            batch (``Any``): The batch. Unused.
            batch_idx (int): The index of the batch. Unused.

        """
        if self._log_stats.inter_step_time:
            self._snap_inter_step_time = time.time()

        if not trainer._logger_connector.should_update_logs:
            return

        gpu_stat_keys = (
            self._get_gpu_stat_keys() + self._get_gpu_device_stat_keys()
        )
        gpu_stats = self._get_gpu_stats([k for k, _ in gpu_stat_keys])
        logs = self._parse_gpu_stats(
            self._device_ids, gpu_stats, gpu_stat_keys
        )

        if self._log_stats.intra_step_time and self._snap_intra_step_time:
            logs["batch_time/intra_step (ms)"] = (
                time.time() - self._snap_intra_step_time
            ) * 1000

        assert trainer.logger is not None
        trainer.logger.log_metrics(logs, step=trainer.global_step)

    @staticmethod
    def _get_gpu_ids(device_ids: list[int]) -> list[str]:
        """Return the physical GPU ID of each logical device index."""
        # All devices if `CUDA_VISIBLE_DEVICES` unset
        default = ",".join(str(i) for i in range(torch.cuda.device_count()))
        cuda_visible_devices: list[str] = os.getenv(
            "CUDA_VISIBLE_DEVICES", default=default
        ).split(",")
        return [
            cuda_visible_devices[device_id].strip() for device_id in device_ids
        ]

    def _get_gpu_stats(self, queries: list[str]) -> list[list[float]]:
        if not queries:
            return []
        """Run nvidia-smi to get the gpu stats."""
        gpu_query = ",".join(queries)
        format = "csv,nounits,noheader"
        gpu_ids = ",".join(self._gpu_ids)
        result = subprocess.run(
            [
                # it's ok to suppress the warning here since we ensure nvidia-smi exists during init
                shutil.which("nvidia-smi"),  # type: ignore
                f"--query-gpu={gpu_query}",
                f"--format={format}",
                f"--id={gpu_ids}",
            ],
            encoding="utf-8",
            capture_output=True,
            check=True,
        )

        def _to_float(x: str) -> float:
            try:
                return float(x)
            except ValueError:
                return 0.0

        return [
            [_to_float(x) for x in s.split(", ")]
            for s in result.stdout.strip().split(os.linesep)
        ]

    @staticmethod
    def _parse_gpu_stats(
        device_ids: list[int],
        stats: list[list[float]],
        keys: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Map the GPU statistics to their metric names."""
        logs = {}
        for i, device_id in enumerate(device_ids):
            for j, (x, unit) in enumerate(keys):
                if unit == "%":
                    unit = "percent"
                logs[f"GPU_{device_id}/{x} - {unit}"] = stats[i][j]
        return logs

    def _get_gpu_stat_keys(self) -> list[tuple[str, str]]:
        """Return the queries and units of the utilization flags."""
        stat_keys = []

        if self._log_stats.gpu_utilization:
            stat_keys.append(("utilization.gpu", "%"))

        if self._log_stats.memory_utilization:
            stat_keys.extend(
                [
                    ("memory.used", "MB"),
                    ("memory.free", "MB"),
                    ("utilization.memory", "%"),
                ]
            )

        return stat_keys

    def _get_gpu_device_stat_keys(self) -> list[tuple[str, str]]:
        """Return the fan and temperature queries with their units."""
        stat_keys = []

        if self._log_stats.fan_speed:
            stat_keys.append(("fan.speed", "%"))

        if self._log_stats.temperature:
            stat_keys.extend(
                [("temperature.gpu", "°C"), ("temperature.memory", "°C")]
            )

        return stat_keys
