import pytest
import torch
import torchvision
import os
import tarfile
import onnx
import json
import yaml

import luxonis_train
from luxonis_train.core import Archiver


class TestArchiver:

    @classmethod
    def setup_class(cls):
        """Create and load all files required for testing."""

        luxonis_train_parent_dir = os.path.dirname(
            os.path.dirname(luxonis_train.__file__)
        )
        tmp_test_path = os.path.join(
            luxonis_train_parent_dir,
            "tests",
            "unittests",
            "test_core",
        )

        # make config
        config_dict = {
            "model": {"name": "dummy", "predefined_model": {"name": "DetectionModel"}},
            "dataset": {
                "name": "dummyldf"
            },  # TODO: set LDF name automatically - choose one random LDF or make a random LDF!
        }
        cls.config_path = os.path.join(tmp_test_path, "tmp_config.yaml")
        with open(cls.config_path, "w") as yaml_file:
            yaml_str = yaml.dump(config_dict)
            yaml_file.write(yaml_str)

        # make model
        model = torchvision.models.squeezenet1_0(pretrained=False)
        cls.model_path = "tmp_squeezenet1_0.onnx"
        n, c, h, w = 1, 3, 224, 224
        input_shape = torch.randn(n, c, h, w)
        cls.input_names = ["TestInput"]
        cls.output_names = ["TestOutput"]

        torch.onnx.export(
            model,
            input_shape,
            cls.model_path,
            verbose=False,
            input_names=cls.input_names,
            output_names=cls.output_names,
        )

        # make archive
        cls.archive_path = Archiver(cls.config_path).archive(cls.model_path)

        # load archive files into memory
        with tarfile.open(cls.archive_path, mode="r") as tar:
            cls.archive_fnames = tar.getnames()  # List all the contents of the tar file
            for fname in cls.archive_fnames:
                if fname.endswith(".json"):
                    json_file = tar.extractfile(fname)
                    cls.json_dict = json.load(json_file)
                elif fname.endswith(".onnx"):
                    onnx_file = tar.extractfile(fname)
                    cls.onnx_model = onnx.load(onnx_file)

    @classmethod
    def teardown_class(cls):
        """Remove all created files."""
        os.remove(cls.archive_path)
        os.remove(cls.config_path)
        os.remove(cls.model_path)

    def test_archive_suffix(self):
        """Test if nn_archive was created."""
        assert self.archive_path.endswith("tar.gz")

    def test_archive_contents(self):
        """Test if nn_archive consists of exactly one JSON and one ONNX file."""
        assert (
            len(self.archive_fnames) == 2
            and any([fname.endswith(".json") for fname in self.archive_fnames])
            and any([fname.endswith(".onnx") for fname in self.archive_fnames])
        )

    def test_onnx(self):
        """Test if archived ONNX model is valid."""
        try:
            onnx.checker.check_model(self.onnx_model, full_check=True)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
            assert False
        else:
            assert True

    def test_config_inputs(self):
        """Test if archived config inputs are valid."""
        config_input_names = []
        for input in self.json_dict["model"]["inputs"]:
            config_input_names.append(input["name"])
        assert set(self.input_names) == set(config_input_names)

    def test_config_outputs(self):
        """Test if archived config outputs are valid."""
        config_output_names = []
        for input in self.json_dict["model"]["outputs"]:
            config_output_names.append(input["name"])
        assert set(self.output_names) == set(config_output_names)
