from pathlib import Path

from luxonis_ml.utils import LuxonisFileSystem

from loguru import logger

if __name__ == "__main__":
    # Hardcode the checkpoint name
    checkpoint_name = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    dest_dir = Path("tests", "data", "checkpoints")
    local_path = dest_dir / checkpoint_name
    if not local_path.exists():
        remote_path = f"gs://luxonis-test-bucket/luxonis-train-test-data/checkpoints/{checkpoint_name}"
        local_path = LuxonisFileSystem.download(remote_path, dest=dest_dir)
        logger.warning("DinoV3 weights downloaded to " + str(local_path))
