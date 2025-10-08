from tests.unittests.test_losses.test_utils import load_checkpoint

if __name__ == "__main__":
    # Hardcode the checkpoint name
    checkpoint = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    load_checkpoint(checkpoint)
