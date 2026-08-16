from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stanford_data_manifest.json"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stanford_dogs"
    / "Images"
)


IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


def load_manifest(
    path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """
    Load the Stanford Dogs model-ready manifest.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise TypeError(
            "Manifest must contain a top-level JSON object."
        )

    records = manifest.get("records")

    if not isinstance(records, list):
        raise TypeError(
            "Manifest field 'records' must be a list."
        )

    return manifest


def build_train_transform() -> transforms.Compose:
    """
    Image preprocessing and augmentation used
    during model training.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.75, 1.0),
            ),
            transforms.RandomHorizontalFlip(
                p=0.5
            ),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """
    Deterministic preprocessing used for
    validation and test images.
    """

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(
                IMAGE_SIZE
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


class StanfordDogsDataset(Dataset):
    """
    PyTorch Dataset backed by our generated
    Stanford Dogs manifest.
    """

    VALID_SPLITS = {
        "train",
        "validation",
        "test",
    }

    def __init__(
        self,
        split: str,
    ) -> None:

        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split '{split}'. "
                f"Expected one of "
                f"{sorted(self.VALID_SPLITS)}."
            )

        manifest = load_manifest()

        self.split = split

        self.records = [
            record
            for record in manifest["records"]
            if record.get("split") == split
        ]

        if not self.records:
            raise ValueError(
                f"No manifest records found "
                f"for split '{split}'."
            )

        if split == "train":
            self.transform = (
                build_train_transform()
            )
        else:
            self.transform = (
                build_eval_transform()
            )

    def __len__(self) -> int:
        """
        Return the number of images in this split.
        """

        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Load and preprocess one image.
        """

        record = self.records[index]

        image_path = (
            IMAGE_ROOT
            / record["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_tensor = self.transform(
                image
            )

        class_index = int(
            record["class_index"]
        )

        return {
            "image": image_tensor,
            "label": torch.tensor(
                class_index,
                dtype=torch.long,
            ),
            "breed_id": record["breed_id"],
            "canonical_name": (
                record["canonical_name"]
            ),
            "image_path": (
                record["image_path"]
            ),
        }


def create_dataloader(
    split: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for one split.
    """

    dataset = StanfordDogsDataset(
        split=split
    )

    shuffle = (
        split == "train"
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )