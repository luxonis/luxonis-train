import os
from pathlib import Path
from typing import Any, Callable, Optional

import albumentations as A
import cv2
import numpy as np
from luxonis_ml.data import Augmentations
from luxonis_ml.data.augmentations.batch_compose import BatchCompose
from luxonis_ml.data.augmentations.batch_transform import BatchBasedTransform
from luxonis_ml.data.augmentations.utils import AUGMENTATIONS
from torch import Tensor
from torch.utils.data import Dataset


def render_visualizations(
    visualizations: dict[str, dict[str, Tensor]],
    save_dir: str | Path | None,
    img_idx: Optional[int] = None,
) -> None:
    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True, parents=True)

    i = 0
    for node_name, vzs in visualizations.items():
        for viz_name, viz_batch in vzs.items():
            for i, viz in enumerate(viz_batch):
                viz_arr = viz.detach().cpu().numpy().transpose(1, 2, 0)
                viz_arr = cv2.cvtColor(viz_arr, cv2.COLOR_RGB2BGR)
                name = f"{node_name}/{viz_name}/{i}"
                if save_dir is not None:
                    name = name.replace("/", "_")
                    if img_idx is not None:
                        cv2.imwrite(
                            str(save_dir / f"{name}_{i}_{img_idx}.png"),
                            viz_arr,
                        )  # img_idx the number of an image in a dir for the inference method
                        i += 1
                    else:
                        cv2.imwrite(str(save_dir / f"{name}_{i}.png"), viz_arr)
                        i += 1
                else:
                    cv2.imshow(name, viz_arr)

    if save_dir is None:
        if cv2.waitKey(0) == ord("q"):
            exit()


class InferDataset(Dataset):
    def __init__(
        self, image_dir: str, augmentations: Optional[Callable] = None
    ):
        """Dataset for using with the infernce method.

        @type image_dir: str
        @param image_dir: Path to the directory with images.
        @type augmentations: Callable | Optional
        @param augmentations: Optional transform to be applied on a
            sample image.
        """
        self.image_dir = image_dir
        self.image_filenames = [
            f
            for f in os.listdir(image_dir)
            if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        self.transform = augmentations

    def __len__(self) -> int:
        """Returns the total number of images."""
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """
        @type idx: int
        @param idx: Index of the image to fetch.
        @rtype: dict[str, Tensor]
        @return: Transformed image.
        """
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform([image])

        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        img_tensor = Tensor(image)

        return {"image": img_tensor}


class InferAugmentations(Augmentations):
    def __init__(
        self,
        image_size: list[int],
        augmentations: list[dict[str, Any]],
        train_rgb: bool = True,
        keep_aspect_ratio: bool = True,
        only_normalize: bool = True,
    ):
        super().__init__(
            image_size,
            augmentations,
            train_rgb,
            keep_aspect_ratio,
            only_normalize,
        )

        (
            self.batch_transform,
            self.spatial_transform,
            self.pixel_transform,
            self.resize_transform,
        ) = self._parse_cfg(
            image_size=image_size,
            augmentations=[
                a for a in augmentations if a["name"] == "Normalize"
            ]
            if only_normalize
            else augmentations,
            keep_aspect_ratio=keep_aspect_ratio,
        )

    def _parse_cfg(
        self,
        image_size: list[int],
        augmentations: list[dict[str, Any]],
        keep_aspect_ratio: bool = True,
    ) -> tuple[BatchCompose, A.Compose, A.Compose, A.Compose]:
        """Parses provided config and returns Albumentations
        BatchedCompose object and Compose object for default transforms.

        @type image_size: List[int]
        @param image_size: Desired image size [H,W]
        @type augmentations: List[Dict[str, Any]]
        @param augmentations: List of augmentations to use and their
            params
        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether should use resize that keeps
            aspect ratio of original image.
        @rtype: Tuple[BatchCompose, A.Compose, A.Compose, A.Compose]
        @return: Objects for batched, spatial, pixel and resize
            transforms
        """

        # NOTE: Always perform Resize
        if keep_aspect_ratio:
            resize = AUGMENTATIONS.get("LetterboxResize")(
                height=image_size[0], width=image_size[1]
            )
        else:
            resize = A.Resize(image_size[0], image_size[1])

        pixel_augs = []
        spatial_augs = []
        batched_augs = []
        if augmentations:
            for aug in augmentations:
                curr_aug = AUGMENTATIONS.get(aug["name"])(
                    **aug.get("params", {})
                )
                if isinstance(curr_aug, A.ImageOnlyTransform):
                    pixel_augs.append(curr_aug)
                elif isinstance(curr_aug, A.DualTransform):
                    spatial_augs.append(curr_aug)
                elif isinstance(curr_aug, BatchBasedTransform):
                    self.is_batched = True
                    self.aug_batch_size = max(
                        self.aug_batch_size, curr_aug.batch_size
                    )
                    batched_augs.append(curr_aug)

        batch_transform = BatchCompose(
            [
                *batched_augs,
            ],
        )

        spatial_transform = A.Compose(
            spatial_augs,
        )

        pixel_transform = A.Compose(
            pixel_augs,
        )

        resize_transform = A.Compose(
            [resize],
        )

        return (
            batch_transform,
            spatial_transform,
            pixel_transform,
            resize_transform,
        )

    def __call__(
        self,
        data: list[np.ndarray],
    ) -> np.ndarray:
        """Performs augmentations on provided data.

        @type data: np.ndarray
        @param data: Data with list of input images and their
            annotations
        @rtype: np.ndarray
        @return: Output image
        """

        image_batch = []

        for img in data:
            image_batch.append(img)

        # Apply batch transform
        transform_args = {
            "image_batch": image_batch,
        }

        transformed = self.batch_transform(force_apply=False, **transform_args)
        transformed = {
            key: np.array(value[0]) for key, value in transformed.items()
        }

        arg_names = [
            "image",
        ]

        # Apply spatial transform
        transformed = self._apply_transform(
            transformed, arg_names, self.spatial_transform, arg_suffix="_batch"
        )

        # Resize if necessary
        if transformed["image"].shape[:2] != self.image_size:
            transformed = self._apply_transform(
                transformed, arg_names, self.resize_transform
            )

        # Apply pixel transform
        transformed = self._apply_transform(
            transformed, arg_names, self.pixel_transform
        )

        out_image = self.post_transform_process(
            transformed,
        )

        return out_image

    def post_transform_process(
        self,
        transformed_data: dict[str, np.ndarray],
    ) -> np.ndarray:
        out_image = transformed_data["image"]
        if not self.train_rgb:
            out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        out_image = out_image.astype(np.float32)
        return out_image
