"""GPU Stats Monitor.

Monitor and logs GPU stats during training.

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
    def __init__(
        self,
        memory_utilization: bool = True,
        gpu_utilization: bool = True,
        intra_step_time: bool = False,
        inter_step_time: bool = False,
        fan_speed: bool = False,
        temperature: bool = False,
    ):
        """Automatically monitors and logs GPU stats during training
        stage. C{GPUStatsMonitor} is a callback and in order to use it
        you need to assign a logger in the C{Trainer}.

        GPU stats are mainly based on C{nvidia-smi --query-gpu} command. The description of the queries is as follows:

            - C{fan.speed} - The fan speed value is the percent of maximum speed that the device's fan is currently
              intended to run at. It ranges from 0 to 100 %. Note: The reported speed is the intended fan speed.
              If the fan is physically blocked and unable to spin, this output will not match the actual fan speed.
              Many parts do not report fan speeds because they rely on cooling via fans in the surrounding enclosure.
            - C{memory.used} - Total memory allocated by active contexts.
            - C{memory.free} - Total free memory.
            - C{utilization.gpu} - Percent of time over the past sample period during which one or more kernels was
              executing on the GPU. The sample period may be between 1 second and 1/6 second depending on the product.
            - C{utilization.memory} - Percent of time over the past sample period during which global (device) memory was
              being read or written. The sample period may be between 1 second and 1/6 second depending on the product.
            - C{temperature.gpu} - Core GPU temperature, in degrees C.
            - C{temperature.memory} - HBM memory temperature, in degrees C.

        @type memory_utilization: bool
        @param memory_utilization: Set to C{True} to monitor used, free and percentage of memory utilization at the start and end of each step. Defaults to C{True}.
        @type gpu_utilization: bool
        @param gpu_utilization: Set to C{True} to monitor percentage of GPU utilization at the start and end of each step. Defaults to C{True}.
        @type intra_step_time: bool
        @param intra_step_time: Set to C{True} to monitor the time of each step. Defaults to {False}.
        @type inter_step_time: bool
        @param inter_step_time: Set to C{True} to monitor the time between the end of one step and the start of the next step. Defaults to C{False}.
        @type fan_speed: bool
        @param fan_speed: Set to C{True} to monitor percentage of fan speed. Defaults to C{False}.
        @type temperature: bool
        @param temperature: Set to C{True} to monitor the memory and gpu temperature in degree Celsius. Defaults to C{False}.
        @raises MisconfigurationException: If NVIDIA driver is not installed, not running on GPUs, or C{Trainer} has no logger.
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
        if shutil.which("nvidia-smi") is None:
            return False
        return CUDAAccelerator.is_available()

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: str | None = None,
    ) -> None:
        if not trainer.logger:
            raise MisconfigurationException(
                "Cannot use GPUStatsMonitor callback with Trainer that has no logger."
            )

        if not CUDAAccelerator.is_available():
            raise MisconfigurationException(
                "You are using GPUStatsMonitor teh CUDA Accelerator is not available."
            )

        # The logical device IDs for selected devices
        # ignoring mypy check because `trainer.data_parallel_device_ids` is None when using CPU
        self._device_ids = sorted(set(trainer.device_ids))

        # The unmasked real GPU IDs
        self._gpu_ids = self._get_gpu_ids(self._device_ids)

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
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
        """Get the unmasked real GPU IDs."""
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
                # it's ok to supress the warning here since we ensure nvidia-smi exists during init
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
        """Parse the gpu stats into a loggable dict."""
        logs = {}
        for i, device_id in enumerate(device_ids):
            for j, (x, unit) in enumerate(keys):
                if unit == "%":
                    unit = "percent"
                logs[f"GPU_{device_id}/{x} - {unit}"] = stats[i][j]
        return logs

    def _get_gpu_stat_keys(self) -> list[tuple[str, str]]:
        """Get the GPU stats keys."""
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
        """Get the device stats keys."""
        stat_keys = []

        if self._log_stats.fan_speed:
            stat_keys.append(("fan.speed", "%"))

        if self._log_stats.temperature:
            stat_keys.extend(
                [("temperature.gpu", "°C"), ("temperature.memory", "°C")]
            )

        return stat_keys
