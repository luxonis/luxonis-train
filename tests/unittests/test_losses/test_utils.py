from pathlib import Path

import torch
from luxonis_ml.utils import LuxonisFileSystem


def load_checkpoint(
    checkpoint_name: str, map_location: torch.device | str | None = None
):
    dest_dir = Path("tests", "data", "checkpoints")
    local_path = dest_dir / checkpoint_name
    if not local_path.exists():
        remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
        local_path = LuxonisFileSystem.download(remote_path, dest=dest_dir)
    return torch.load(local_path, map_location=map_location)
