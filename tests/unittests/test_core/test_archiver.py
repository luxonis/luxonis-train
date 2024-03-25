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
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.nn_archive.config_building_blocks.base_models import head_outputs
from parameterized import parameterized

import luxonis_train
from luxonis_train.core import Archiver
from luxonis_train.core.exporter import Exporter
from luxonis_train.core.trainer import Trainer
from luxonis_train.nodes.enums.head_categorization import ImplementedHeads
from luxonis_train.utils.config import Config

HEAD_NAMES = [head_name for head_name in ImplementedHeads.__members__]


class TestArchiver:
    @classmethod
    def setup_class(cls):
        """Creates all files required for testing."""

        # make tmp dir
        luxonis_train_parent_dir = os.path.dirname(
            os.path.dirname(luxonis_train.__file__)
        )
        cls.tmp_path = os.path.join(
            luxonis_train_parent_dir, "tests", "unittests", "test_core", "tmp"
        )
        os.mkdir(cls.tmp_path)

        # make LDFs
        unilabelLDF = "dummyLDF_unilabel"
        cls._make_dummy_ldf(
            ldf_name=unilabelLDF,
            save_path=cls.tmp_path,
            bbx_anno=True,
            kpt_anno=True,
        )
        multilabelLDF = "dummyLDF_multilabel"
        cls._make_dummy_ldf(
            ldf_name=multilabelLDF,
            save_path=cls.tmp_path,
            cls_anno=True,
            bbx_anno=True,
            sgm_anno=True,
            multilabel=True,
        )
        cls.ldf_names = [unilabelLDF, multilabelLDF]

        for head_name in HEAD_NAMES:
            if head_name == "ImplicitKeypointBBoxHead":
                ldf_name = unilabelLDF  # multiclass keypoint detection not yet supported in luxonis-train
            else:
                ldf_name = multilabelLDF

            # make config
            cfg_dict = cls._make_dummy_cfg_dict(
                head_name=head_name,
                save_path=cls.tmp_path,
                ldf_name=ldf_name,
            )
            cfg = Config.get_config(cfg_dict)

            # train model
            cfg.trainer.epochs = 1
            cfg.trainer.validation_interval = 1
            cfg.trainer.batch_size = 1
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
            cfg.archiver.archive_name = f"nnarchive_{head_name}"
            archiver = Archiver(cfg=cfg)
            cls.archive_path = archiver.archive(cls.onnx_model_path)
            archiver.reset_logging()

            # clear the loaded config instance
            Config.clear_instance()

    def _make_dummy_ldf(
        ldf_name: str,
        save_path: str,
        number: int = 3,
        dim: tuple = (10, 10, 3),
        cls_anno: bool = False,
        bbx_anno: bool = False,
        sgm_anno: bool = False,
        kpt_anno: bool = False,
        multilabel: bool = False,
        split_ratios: list = None,
    ):
        """Creates random-pixel images with fictional annotations and parses them to
        L{LuxonisDataset} format.

        @type ldf_name: str
        @param ldf_name: Name of the created L{LuxonisDataset} format dataset.
        @type save_path: str
        @param save_path: Path to where the created images are saved.
        @type number: int
        @param number: Number of images to create.
        @type dim: Tuple[int, int, int]
        @param dim: Dimensions of the created images in HWC order.
        @type cls_anno: bool
        @param cls_anno: True if created dataset should contain classification annotations.
        type bbx_anno: bool
        @param bbx_anno: True if created dataset should contain bounding box annotations.
        type sgm_anno: bool
        @param sgm_anno: True if created dataset should contain segmentation annotations.
        type kpt_anno: bool
        @param kpt_anno: True if created dataset should contain keypoint annotations.
        type multilabel: bool
        @param multilabel: True if created dataset should contain multilabel annotations.
        type split_ratios: List[float, float, float]
        @param split_ratios: List of ratios defining the train, val, and test splits.
        """

        if split_ratios is None:
            split_ratios = [0.333, 0.333, 0.333]

        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)

        if multilabel:
            labels = ["label_x", "label_y", "label_z"]
        else:
            labels = ["label_x"]

        def dataset_generator():
            for i in range(number):
                label = random.choice(labels)
                img = np.random.randint(0, 256, dim, dtype=np.uint8)
                img_file_path = os.path.join(save_path, "images", f"img{i}.png")
                cv2.imwrite(img_file_path, img)

                if cls_anno:
                    yield {
                        "file": img_file_path,
                        "type": "classification",
                        "value": True,
                        "class": label,
                    }

                if bbx_anno:
                    box = (0.25, 0.25, 0.5, 0.5)
                    yield {
                        "file": img_file_path,
                        "type": "box",
                        "value": box,
                        "class": label,
                    }

                if kpt_anno:
                    keypoints = [
                        (0.25, 0.25, 2),
                        (0.75, 0.25, 2),
                        (0.75, 0.75, 2),
                        (0.25, 0.75, 2),
                    ]
                    yield {
                        "file": img_file_path,
                        "type": "keypoints",
                        "value": keypoints,
                        "class": label,
                    }

                if sgm_anno:
                    polyline = [
                        (0.25, 0.75),
                        (0.75, 0.25),
                        (0.75, 0.75),
                        (0.25, 0.75),
                        (0.25, 0.25),
                    ]
                    yield {
                        "file": img_file_path,
                        "type": "polyline",
                        "value": polyline,
                        "class": label,
                    }

        if LuxonisDataset.exists(ldf_name):
            print("Deleting existing dataset")
            LuxonisDataset(ldf_name).delete_dataset()
        dataset = LuxonisDataset(ldf_name)
        dataset.set_classes(list(labels))
        if kpt_anno:
            keypoint_labels = [
                "kp1",
                "kp2",
                "kp3",
                "kp4",
            ]
            keypoint_edges = [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ]
            dataset.set_skeletons(
                {
                    label: {"labels": keypoint_labels, "edges": keypoint_edges}
                    for label in labels
                }
            )
        dataset.add(dataset_generator)
        dataset.make_splits(ratios=split_ratios)

    def _make_dummy_cfg_dict(head_name: str, ldf_name: str, save_path: str) -> dict:
        """Creates a configuration dict based on the type of the provided model head.

        @type head_name: str
        @param head_name: Name of the specified head.
        @type ldf_name: str
        @param ldf_name: Name of the L{LuxonisDataset} format dataset on which the
            training will be performed.
        @type save_path: str
        @param save_path: Path to LuxonisTrackerPL save directory.
        @rtype: dict
        @return: Created config dict.
        """

        cfg_dict = {"model": {"name": f"model_w_{head_name}"}}
        cfg_dict["dataset"] = {"name": ldf_name}
        cfg_dict["tracker"] = {"save_directory": save_path}

        if head_name == "ClassificationHead":
            cfg_dict["model"]["predefined_model"] = {"name": "ClassificationModel"}
        elif head_name == "EfficientBBoxHead":
            cfg_dict["model"]["predefined_model"] = {"name": "DetectionModel"}
        elif head_name == "ImplicitKeypointBBoxHead":
            cfg_dict["model"]["predefined_model"] = {"name": "KeypointDetectionModel"}
        elif head_name == "SegmentationHead":
            cfg_dict["model"]["predefined_model"] = {"name": "SegmentationModel"}
        elif head_name == "BiSeNetHead":
            cfg_dict["model"]["nodes"] = [
                {"name": "MicroNet", "alias": "segmentation_backbone"},
                {
                    "name": "BiSeNetHead",
                    "alias": "segmentation_head",
                    "inputs": ["segmentation_backbone"],
                },
            ]
            cfg_dict["model"]["losses"] = [
                {"name": "BCEWithLogitsLoss", "attached_to": "segmentation_head"}
            ]
        else:
            raise NotImplementedError(f"No implementation for {head_name}")

        return cfg_dict

    @parameterized.expand(HEAD_NAMES)
    def test_archive_creation(self, head_name):
        """Tests if NN archive was created using xz compression (should be the default
        option)."""
        archive_path = os.path.join(self.tmp_path, f"nnarchive_{head_name}_onnx.tar.xz")
        assert archive_path.endswith("tar.xz")

    @parameterized.expand(HEAD_NAMES)
    def test_archive_contents(self, head_name):
        """Tests if NN archive consists of config.json and model.onnx."""
        archive_path = os.path.join(self.tmp_path, f"nnarchive_{head_name}_onnx.tar.xz")
        with tarfile.open(archive_path, mode="r") as tar:
            archive_fnames = tar.getnames()
        assert (
            len(archive_fnames) == 2
            and any([fname == "config.json" for fname in archive_fnames])
            and any([fname == "model.onnx" for fname in archive_fnames])
        )

    @parameterized.expand(HEAD_NAMES)
    def test_onnx(self, head_name):
        """Tests if archive ONNX model is valid."""
        archive_path = os.path.join(self.tmp_path, f"nnarchive_{head_name}_onnx.tar.xz")
        with tarfile.open(archive_path, mode="r") as tar:
            f = tar.extractfile("model.onnx")
            model_bytes = f.read()
            model_io = io.BytesIO(model_bytes)
            onnx_model = onnx.load(model_io)
        assert onnx.checker.check_model(onnx_model, full_check=True) is None

    @parameterized.expand(HEAD_NAMES)
    def test_config_io(self, head_name):
        """Tests if archived config inputs and outputs are valid."""
        archive_path = os.path.join(self.tmp_path, f"nnarchive_{head_name}_onnx.tar.xz")
        with tarfile.open(archive_path, mode="r") as tar:
            f = tar.extractfile("config.json")
            json_dict = json.load(f)
            f = tar.extractfile("model.onnx")
            model_bytes = f.read()
            model_io = io.BytesIO(model_bytes)
            onnx_model = onnx.load(model_io)

        config_input_names = []
        for input in json_dict["model"]["inputs"]:
            config_input_names.append(input["name"])
        valid_inputs = set([input.name for input in onnx_model.graph.input]) == set(
            config_input_names
        )

        config_output_names = []
        for input in json_dict["model"]["outputs"]:
            config_output_names.append(input["name"])
        valid_outputs = set([output.name for output in onnx_model.graph.output]) == set(
            config_output_names
        )

        assert valid_inputs and valid_outputs

    @parameterized.expand(HEAD_NAMES)
    def test_head_outputs(self, head_name):
        """Tests if archived config head outputs are valid."""
        archive_path = os.path.join(self.tmp_path, f"nnarchive_{head_name}_onnx.tar.xz")
        with tarfile.open(archive_path, mode="r") as tar:
            f = tar.extractfile("config.json")
            json_dict = json.load(f)
        head_output = json_dict["model"]["heads"][0]["outputs"]
        if head_name == "ClassificationHead":
            assert head_outputs.OutputsClassification.parse_obj(head_output)
        elif head_name == "EfficientBBoxHead":
            assert head_outputs.OutputsYOLO.parse_obj(head_output)
        elif head_name == "ImplicitKeypointBBoxHead":
            assert head_outputs.OutputsKeypointDetectionYOLO.parse_obj(head_output)
        elif head_name == "SegmentationHead":
            assert head_outputs.OutputsSegmentation.parse_obj(head_output)
        elif head_name == "BiSeNetHead":
            assert head_outputs.OutputsSegmentation.parse_obj(head_output)
        else:
            raise NotImplementedError(f"Missing tests for {head_name} head")

    @classmethod
    def teardown_class(cls):
        """Removes all files created during setup."""
        for ldf_name in cls.ldf_names:
            LuxonisDataset(ldf_name).delete_dataset()
        shutil.rmtree(cls.tmp_path)
