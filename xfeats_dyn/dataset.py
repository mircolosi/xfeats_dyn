import json
import os
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import torch
import typer
from lightning import LightningDataModule
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Tuple

from xfeats_dyn.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

MOVABLE_SUPERCATEGORY = dict(enumerate(["other", "person", "sports", "animal", "vehicle"]))
MOVABLE_SUPERCATEGORY_IDS = {v: k for k, v in MOVABLE_SUPERCATEGORY.items()}
W = 800
H = 600


class Coco2017Dataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, stage=None, keep_n=None):
        """
        Args:
            root_dir (string): Directory where COCO17 is stored.
            annotation_file (string): Path to the COCO json annotation file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        if annotation_file is None:
            raise NotImplementedError("Test case still needs to be implemented.")

        with open(annotation_file, "r") as f:
            self.coco_annotations = json.load(f)
        self.annotations = self.coco_annotations["annotations"][:keep_n]
        self.images = self.coco_annotations["images"][:keep_n]
        self.image_id_to_filename = {img["id"]: img["file_name"] for img in self.images}
        self.image_id_to_annotations = {}
        self.class_id_to_supercategory = {d["id"]: d["supercategory"] for d in self.coco_annotations["categories"]}

        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

        self.stage = stage if stage is not None else "debug"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[Image.Image, dict]:
        image_data = self.images[idx]
        image_id = image_data["id"]
        annotation = self.image_id_to_annotations.get(image_id, [])
        image_file = os.path.join(self.root_dir, image_data["file_name"])

        raw_image = Image.open(image_file)

        image = raw_image
        if self.transform:
            image = self.transform(raw_image)

        masks = np.zeros(
            (
                image_data["height"],
                image_data["width"],
                len(MOVABLE_SUPERCATEGORY),
            ),
            dtype=np.float32,
        )
        for ann in annotation:
            seg = ann["segmentation"]

            if isinstance(seg, dict):  # RLE format
                seg = [ann["segmentation"]["counts"]]
            if isinstance(seg, list):  # polygon format
                pass
            else:  # RLE format
                raise NotImplementedError(f"RLE format not implemented yet.\n{ann['segmentation']}")
            rle = mask_utils.frPyObjects(
                seg,
                image_data["height"],
                image_data["width"],
            )
            m = mask_utils.decode(rle)
            m = np.sum(m, axis=2)
            # binary_mask = (m > 100).astype(np.float32)
            category = self.class_id_to_supercategory[ann["category_id"]]
            if category not in MOVABLE_SUPERCATEGORY_IDS:
                category = "other"
            category_idx = MOVABLE_SUPERCATEGORY_IDS[category]
            masks[:, :, category_idx] += m

        masks = (masks > 0.1).astype(np.float32)
        seg_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((H, W)),
            ]
        )
        segmentation_mask = seg_transform(masks)
        squashed_mask = ~torch.any(segmentation_mask[1:, :, :], dim=0).unsqueeze(0)

        if self.stage == "debug":
            import matplotlib.pyplot as plt

            _, ax = plt.subplots(1, masks.shape[-1] + 2, figsize=(15, 15))
            for i in range(masks.shape[-1]):
                # print(i, MOVABLE_SUPERCATEGORY[i], segmentation_mask[i, :, :].shape)
                ax[i].imshow(segmentation_mask[i, :, :], cmap="gray")
                ax[i].axis("off")
                ax[i].set_title(MOVABLE_SUPERCATEGORY[i])
            logger.info(f"{segmentation_mask[1:, :, :].shape=}")
            ax[-2].imshow(squashed_mask[0, ...], cmap="gray")
            ax[-2].axis("off")
            ax[-2].set_title("Tokeep mask")
            ax[-1].imshow(image.permute(1, 2, 0), cmap="gray")
            ax[-1].axis("off")
            ax[-1].set_title("Image")
            plt.show()
            logger.info(f"{image_file} [{idx=}], {squashed_mask.shape=}")

        label = {
            "segmentation_mask": squashed_mask,
            "image_id": image_id,
        }

        return image, label


class Coco2017DataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()

        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.transform = v2.Compose(
            [
                v2.Grayscale(num_output_channels=1),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((H, W)),  # Example resize
                v2.Normalize(mean=[0.44531356896770125], std=[0.2692461874154524]),  # Standard ImageNet normalization
            ]
        )

    def setup(self, stage):
        print(f"Setting up stage {stage}...")

        if stage in ["fit", "validate"]:
            self.val_dataset = Coco2017Dataset(
                root_dir=os.path.join(RAW_DATA_DIR, "2017", "val2017"),
                annotation_file=os.path.join(
                    RAW_DATA_DIR,
                    "2017",
                    "annotations",
                    "instances_val2017.json",
                ),
                transform=self.transform,
                stage=stage,
            )

            if stage == "fit":
                self.train_dataset = Coco2017Dataset(
                    root_dir=os.path.join(RAW_DATA_DIR, "2017", "train2017"),
                    annotation_file=os.path.join(
                        RAW_DATA_DIR,
                        "2017",
                        "annotations",
                        "instances_train2017.json",
                    ),
                    transform=self.transform,
                    stage=stage,
                )

        elif stage == "test":
            self.test_dataset = Coco2017Dataset(
                root_dir=os.path.join(RAW_DATA_DIR, "2017", "test2017"),
                annotation_file=None,
                transform=self.transform,
                stage=stage,
            )
        elif stage == "predict":
            self.predict_dataset = Coco2017Dataset(
                root_dir=os.path.join(RAW_DATA_DIR, "2017", "train2017"),
                annotation_file=os.path.join(
                    RAW_DATA_DIR,
                    "2017",
                    "annotations",
                    "instances_train2017.json",
                ),
                transform=self.transform,
                stage=stage,
                keep_n=32,
            )

        else:
            return
            raise NotImplementedError(f"Stage {stage} not implemented.")

    def train_dataloader(self):
        logger.info("Loading train dataloader...")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
        )

    def val_dataloader(self):
        logger.info("Loading val dataloader...")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def test_dataloader(self):
        logger.info("Loading test dataloader...")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
        )


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
