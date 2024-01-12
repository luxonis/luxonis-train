import argparse
import os

import cv2
import torch
from luxonis_ml.data import (
    LuxonisDataset,
    TrainAugmentations,
    ValAugmentations,
)

from luxonis_train.attached_modules.visualizers.utils import (
    draw_bounding_box_labels,
    draw_keypoint_labels,
    draw_segmentation_labels,
    get_unnormalized_images,
)
from luxonis_train.utils.config import Config
from luxonis_train.utils.loaders import LuxonisLoaderTorch, collate_fn
from luxonis_train.utils.types import LabelType

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration file to use",
    )
    parser.add_argument(
        "--view",
        type=str,
        default="val",
        help="Dataset view to use",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display images",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Path to save directory, by default don't save",
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional options")
    args = parser.parse_args()

    opts = args.opts or []
    overrides = {}
    if opts:
        if len(opts) % 2 != 0:
            raise ValueError("Override options should be a list of key-value pairs")
        for i in range(0, len(opts), 2):
            overrides[opts[i]] = opts[i + 1]

    cfg = Config.get_config(args.config, overrides)

    image_size = cfg.trainer.preprocessing.train_image_size

    dataset = LuxonisDataset(
        dataset_name=cfg.dataset.dataset_name,
        team_id=cfg.dataset.team_id,
        dataset_id=cfg.dataset.dataset_id,
        bucket_type=cfg.dataset.bucket_type,
        bucket_storage=cfg.dataset.bucket_storage,
    )
    augmentations = (
        TrainAugmentations(
            image_size=image_size,
            augmentations=[
                i.model_dump() for i in cfg.trainer.preprocessing.augmentations
            ],
            train_rgb=cfg.trainer.preprocessing.train_rgb,
            keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
        )
        if args.view == "train"
        else ValAugmentations(
            image_size=image_size,
            augmentations=[
                i.model_dump() for i in cfg.trainer.preprocessing.augmentations
            ],
            train_rgb=cfg.trainer.preprocessing.train_rgb,
            keep_aspect_ratio=cfg.trainer.preprocessing.keep_aspect_ratio,
        )
    )

    loader_train = LuxonisLoaderTorch(
        dataset,
        view=args.view,
        augmentations=augmentations,
    )

    pytorch_loader_train = torch.utils.data.DataLoader(
        loader_train,
        batch_size=4,
        num_workers=1,
        collate_fn=collate_fn,
    )

    save_dir = args.save_dir
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    counter = 0
    for data in pytorch_loader_train:
        imgs, label_dict = data
        images = get_unnormalized_images(cfg, imgs)
        for i, img in enumerate(images):
            for label_type, labels in label_dict.items():
                if label_type == LabelType.CLASSIFICATION:
                    continue
                elif label_type == LabelType.BOUNDINGBOX:
                    img = draw_bounding_box_labels(
                        img, labels[labels[:, 0] == i][:, 2:], colors="yellow", width=1
                    )
                elif label_type == LabelType.KEYPOINT:
                    img = draw_keypoint_labels(
                        img, labels[labels[:, 0] == i][:, 1:], colors="red"
                    )
                elif label_type == LabelType.SEGMENTATION:
                    img = draw_segmentation_labels(
                        img, labels[i], alpha=0.8, colors="#5050FF"
                    )

            img_arr = img.permute(1, 2, 0).numpy()
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
            if save_dir is not None:
                counter += 1
                cv2.imwrite(os.path.join(save_dir, f"{counter}.png"), img_arr)
            if not args.no_display:
                cv2.imshow("img", img_arr)
                if cv2.waitKey() == ord("q"):
                    exit()
