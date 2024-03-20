import io
import json
import os
import random
import shutil
import tarfile

import cv2
import lightning.pytorch as pl
import numpy as np
import onnx
import yaml
from luxonis_ml.data import LuxonisDataset

import luxonis_train
from luxonis_train.core import Archiver
from luxonis_train.core.exporter import Exporter
from luxonis_train.core.trainer import Trainer
from luxonis_train.utils.config import Config


class TestArchiver:
    @classmethod
    def setup_class(cls):
        """Create and load all files required for testing."""

        luxonis_train_parent_dir = os.path.dirname(
            os.path.dirname(luxonis_train.__file__)
        )
        cls.tmp_path = os.path.join(
            luxonis_train_parent_dir, "tests", "unittests", "test_core", "tmp"
        )
        os.mkdir(cls.tmp_path)

        # make LDF
        os.mkdir(os.path.join(cls.tmp_path, "images"))
        cls.ldf_name = "dummyLDF"
        labels = ["label1", "label2", "label3"]

        def classification_dataset_generator():
            for i in range(10):
                img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
                img_file_path = os.path.join(cls.tmp_path, "images", f"img{i}.png")
                cv2.imwrite(img_file_path, img)
                yield {
                    "file": img_file_path,
                    "type": "classification",
                    "value": True,
                    "class": random.choice(labels),
                }

        if LuxonisDataset.exists(cls.ldf_name):
            print("Deleting existing dataset")
            LuxonisDataset(cls.ldf_name).delete_dataset()
        dataset = LuxonisDataset(cls.ldf_name)
        dataset.add(classification_dataset_generator)
        dataset.set_classes(list(labels))
        dataset.make_splits()

        # make config
        config_dict = {
            "model": {
                "name": "test_model",
                "predefined_model": {"name": "ClassificationModel"},
            },
            "dataset": {"name": cls.ldf_name},
            "tracker": {"save_directory": cls.tmp_path},
        }
        cls.config_path = os.path.join(cls.tmp_path, "config.yaml")
        with open(cls.config_path, "w") as yaml_file:
            yaml_str = yaml.dump(config_dict)
            yaml_file.write(yaml_str)
        cfg = Config.get_config(config_dict)

        # train model
        cfg.trainer.epochs = 1
        cfg.trainer.validation_interval = 1
        cfg.trainer.batch_size = 4
        trainer = Trainer(cfg=cfg)
        trainer.train()
        callbacks = [
            c
            for c in trainer.pl_trainer.callbacks
            if isinstance(c, pl.callbacks.ModelCheckpoint)
        ]
        model_checkpoint_path = callbacks[0].best_model_path
        model_ckpt = os.path.join(trainer.run_save_dir, model_checkpoint_path)
        trainer.reset_logging()

        # export model to ONNX
        cfg.model.weights = model_ckpt
        exporter = Exporter(cfg=cfg)
        cls.onnx_model_path = os.path.join(cls.tmp_path, "model.onnx")
        exporter.export(onnx_path=cls.onnx_model_path)
        exporter.reset_logging()

        # make archive
        cfg.archiver.archive_save_directory = cls.tmp_path
        archiver = Archiver(cls.config_path)
        cls.archive_path = archiver.archive(cls.onnx_model_path)
        archiver.reset_logging()

        # load archive files into memory
        with tarfile.open(cls.archive_path, mode="r") as tar:
            cls.archive_fnames = tar.getnames()
            for fname in cls.archive_fnames:
                f = tar.extractfile(fname)
                if fname.endswith(".json"):
                    cls.json_dict = json.load(f)
                elif fname.endswith(".onnx"):
                    model_bytes = f.read()
                    model_io = io.BytesIO(model_bytes)
                    cls.onnx_model = onnx.load(model_io)

    @classmethod
    def teardown_class(cls):
        """Remove all created files."""
        LuxonisDataset(cls.ldf_name).delete_dataset()
        shutil.rmtree(cls.tmp_path)

    def test_archive_creation(self):
        """Test if nn_archive was created."""
        assert os.path.exists(self.archive_path)

    def test_archive_suffix(self):
        """Test if nn_archive is compressed using xz option (should be the default
        option)."""
        assert self.archive_path.endswith("tar.xz")

    def test_archive_contents(self):
        """Test if nn_archive consists of config.json and model.onnx."""
        assert (
            len(self.archive_fnames) == 2
            and any([fname == "config.json" for fname in self.archive_fnames])
            and any([fname == "model.onnx" for fname in self.archive_fnames])
        )

    def test_onnx(self):
        """Test if archived ONNX model is valid."""
        assert onnx.checker.check_model(self.onnx_model, full_check=True) is None

    def test_config_inputs(self):
        """Test if archived config inputs are valid."""
        config_input_names = []
        for input in self.json_dict["model"]["inputs"]:
            config_input_names.append(input["name"])
        assert set([input.name for input in self.onnx_model.graph.input]) == set(
            config_input_names
        )

    def test_config_outputs(self):
        """Test if archived config outputs are valid."""
        config_output_names = []
        for input in self.json_dict["model"]["outputs"]:
            config_output_names.append(input["name"])
        assert set([output.name for output in self.onnx_model.graph.output]) == set(
            config_output_names
        )
