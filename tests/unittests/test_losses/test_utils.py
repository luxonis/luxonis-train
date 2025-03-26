import os

import torch
from luxonis_ml.utils import LuxonisFileSystem


def load_checkpoint(checkpoint_name: str, map_location=None):
    dest_dir = "./tests/data/checkpoints"
    local_path = os.path.join(dest_dir, checkpoint_name)
    if not os.path.exists(local_path):
        remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
        local_path = LuxonisFileSystem.download(remote_path, dest=dest_dir)
    checkpoint = torch.load(local_path, map_location=map_location)
    return checkpoint
