import glob
import json
import os
import zipfile
from pathlib import Path

import cv2
import gdown
import numpy as np
import pytest
import torchvision
from luxonis_ml.data import LuxonisDataset
from luxonis_ml.utils import environ

Path(environ.LUXONISML_BASE_PATH).mkdir(exist_ok=True)


def create_dataset(name: str) -> LuxonisDataset:
    if LuxonisDataset.exists(name):
        dataset = LuxonisDataset(name)
        dataset.delete_dataset()
    return LuxonisDataset(name)


@pytest.fixture(scope="session", autouse=True)
def create_coco_dataset():
    dataset = create_dataset("coco_test")
    url = "https://drive.google.com/uc?id=1XlvFK7aRmt8op6-hHkWVKIJQeDtOwoRT"
    output_zip = "../data/COCO_people_subset.zip"
    output_folder = "../data/"

    if not os.path.exists(output_zip) and not os.path.exists(
        os.path.join(output_folder, "COCO_people_subset")
    ):
        gdown.download(url, output_zip, quiet=False)

        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(output_folder)

    def COCO_people_subset_generator():
        img_dir = "../data/person_val2017_subset"
        annot_file = "../data/person_keypoints_val2017.json"
        im_paths = glob.glob(img_dir + "/*.jpg")
        nums = np.array([int(Path(path).stem) for path in im_paths])
        idxs = np.argsort(nums)
        im_paths = list(np.array(im_paths)[idxs])
        with open(annot_file) as file:
            data = json.load(file)
        imgs = data["images"]
        anns = data["annotations"]

        for path in im_paths:
            gran = Path(path).name
            img = [img for img in imgs if img["file_name"] == gran][0]
            img_id = img["id"]
            img_anns = [ann for ann in anns if ann["image_id"] == img_id]

            im = cv2.imread(path)
            height, width, _ = im.shape

            if len(img_anns):
                yield {
                    "file": path,
                    "class": "person",
                    "type": "classification",
                    "value": True,
                }

            for ann in img_anns:
                seg = ann["segmentation"]
                if isinstance(seg, list):
                    poly = []
                    for s in seg:
                        poly_arr = np.array(s).reshape(-1, 2)
                        poly += [
                            (poly_arr[i, 0] / width, poly_arr[i, 1] / height)
                            for i in range(len(poly_arr))
                        ]
                    yield {
                        "file": path,
                        "class": "person",
                        "type": "polyline",
                        "value": poly,
                    }

                x, y, w, h = ann["bbox"]
                yield {
                    "file": path,
                    "class": "person",
                    "type": "box",
                    "value": (x / width, y / height, w / width, h / height),
                }

                kps = np.array(ann["keypoints"]).reshape(-1, 3)
                keypoint = []
                for kp in kps:
                    keypoint.append(
                        (float(kp[0] / width), float(kp[1] / height), int(kp[2]))
                    )
                yield {
                    "file": path,
                    "class": "person",
                    "type": "keypoints",
                    "value": keypoint,
                }

    dataset.set_classes(["person"])

    annot_file = "../data/person_keypoints_val2017.json"
    with open(annot_file) as file:
        data = json.load(file)
    dataset.set_skeletons(
        {
            "person": {
                "labels": data["categories"][0]["keypoints"],
                "edges": (np.array(data["categories"][0]["skeleton"]) - 1).tolist(),
            }
        }
    )
    dataset.add(COCO_people_subset_generator)  # type: ignore
    dataset.make_splits()


@pytest.fixture(scope="session", autouse=True)
def create_cifar10_dataset():
    dataset = create_dataset("cifar10_test")
    cifar10_torch = torchvision.datasets.CIFAR10(
        root="../data", train=False, download=True
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
            path = f"../data/cifar_{i}.png"
            image.save(path)
            yield {
                "file": path,
                "class": classes[label],
                "type": "classification",
                "value": True,
            }

    dataset.set_classes(classes)

    dataset.add(CIFAR10_subset_generator)  # type: ignore
    dataset.make_splits()
