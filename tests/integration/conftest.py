import os
from pathlib import Path

import gdown
import pytest
import torchvision
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.utils import environ

Path(environ.LUXONISML_BASE_PATH).mkdir(exist_ok=True)


def create_dataset(name: str) -> LuxonisDataset:
    if LuxonisDataset.exists(name):
        dataset = LuxonisDataset(name)
        dataset.delete_dataset()
    return LuxonisDataset(name)


@pytest.fixture(scope="session", autouse=True)
def create_coco_dataset():
    dataset_name = "coco_test"
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    output_folder = "../data/"
    output_zip = os.path.join(output_folder, "COCO_people_subset.zip")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_zip) and not os.path.exists(
        os.path.join(output_folder, "COCO_people_subset")
    ):
        gdown.download(url, output_zip, quiet=False)

    parser = LuxonisParser(output_zip, dataset_name=dataset_name, delete_existing=True)
    parser.parse(random_split=True)


def _create_cifar10(dataset_name: str, task_name: str) -> None:
    dataset = create_dataset(dataset_name)
    output_folder = "../data/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cifar10_torch = torchvision.datasets.CIFAR10(
        root=output_folder, train=False, download=True
    )
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def CIFAR10_subset_generator():
        for i, (image, label) in enumerate(cifar10_torch):  # type: ignore
            if i == 1000:
                break
            path = os.path.join(output_folder, f"cifar_{i}.png")
            image.save(path)
            yield {
                "file": path,
                "annotation": {
                    "type": "classification",
                    "task": task_name,
                    "class": classes[label],
                },
            }

    dataset.add(CIFAR10_subset_generator())
    dataset.make_splits()


@pytest.fixture(scope="session", autouse=True)
def create_cifar10_dataset():
    _create_cifar10("cifar10_test", "classification")


@pytest.fixture(scope="session", autouse=True)
def create_cifar10_task_dataset():
    _create_cifar10("cifar10_task_test", "cifar10_task")
