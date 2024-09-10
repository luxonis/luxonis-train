from pathlib import Path

from luxonis_train.core import LuxonisModel

TEST_OUTPUT = Path("./probe")
OPTS = {
    "trainer.epochs": 1,
    "trainer.batch_size": 4,
    "trainer.validation_interval": 1,
    "trainer.callbacks": "[]",
    "tracker.save_directory": str(TEST_OUTPUT),
    "tuner.n_trials": 4,
}


def main():
    config_file = "obb_detection_model"
    config_file = f"configs/{config_file}.yaml"
    # model = LuxonisModel(config_file, opts=OPTS)
    model = LuxonisModel(config_file)
    model.train()
    model.test()


if __name__ == "__main__":
    main()
