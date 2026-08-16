from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANFORD_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stanford_dogs"
    / "file_list.mat"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stanford_dogs_labels.json"
)


def extract_matlab_string(value: Any) -> str:
    """
    Extract a Python string from Stanford Dogs'
    nested MATLAB/NumPy representation.

    Stanford entries are typically shaped like:

        [array(['n02085620-Chihuahua/example.jpg'])]

    We unwrap NumPy arrays until we reach the actual
    Python/NumPy string value.
    """

    current = value

    while hasattr(current, "ndim") and current.ndim > 0:
        if current.size != 1:
            raise ValueError(
                "Expected a single MATLAB string value, "
                f"but received an array with size {current.size}."
            )

        current = current.item()

    if not isinstance(current, str):
        current = str(current)

    return current.strip()


def parse_breed_directory(
    file_path: str,
) -> tuple[str, str]:
    """
    Parse a Stanford Dogs image path.

    Example:

        n02085620-Chihuahua/n02085620_10074.jpg

    Returns:

        ("n02085620", "Chihuahua")
    """

    breed_directory = file_path.split("/", maxsplit=1)[0]

    match = re.fullmatch(
        r"(n\d+)-(.+)",
        breed_directory,
    )

    if match is None:
        raise ValueError(
            f"Unexpected Stanford breed directory: {breed_directory}"
        )

    imagenet_id = match.group(1)
    raw_breed_name = match.group(2)

    human_name = (
        raw_breed_name
        .replace("_", " ")
        .strip()
    )

    return imagenet_id, human_name


def canonicalize_breed_id(
    breed_name: str,
) -> str:
    """
    Convert a human breed name into a canonical snake_case ID.

    Example:

        "German Shepherd" -> "german_shepherd"
    """

    value = breed_name.lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def extract_breed_records() -> list[dict[str, Any]]:
    """
    Read Stanford Dogs metadata and build one record per breed.
    """

    if not STANFORD_METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Stanford metadata not found: "
            f"{STANFORD_METADATA_PATH}"
        )

    data = loadmat(STANFORD_METADATA_PATH)

    file_list = data["file_list"]
    labels = data["labels"]

    if len(file_list) != len(labels):
        raise ValueError(
            "Stanford file_list and labels have different lengths."
        )

    records_by_numeric_label: dict[int, dict[str, Any]] = {}

    for file_entry, label_entry in zip(
        file_list,
        labels,
        strict=True,
    ):
        file_path = extract_matlab_string(file_entry)

        numeric_label = int(
            label_entry.flat[0]
        )

        imagenet_id, breed_name = parse_breed_directory(
            file_path
        )

        breed_id = canonicalize_breed_id(
            breed_name
        )

        existing = records_by_numeric_label.get(
            numeric_label
        )

        if existing is None:
            records_by_numeric_label[numeric_label] = {
                "stanford_label": numeric_label,
                "breed_id": breed_id,
                "canonical_name": breed_name,
                "imagenet_id": imagenet_id,
                "source_directory": (
                    file_path.split("/", maxsplit=1)[0]
                ),
                "image_count": 1,
            }

        else:
            if existing["imagenet_id"] != imagenet_id:
                raise ValueError(
                    "Numeric label maps to multiple ImageNet IDs: "
                    f"{numeric_label}"
                )

            if existing["breed_id"] != breed_id:
                raise ValueError(
                    "Numeric label maps to multiple breed names: "
                    f"{numeric_label}"
                )

            existing["image_count"] += 1

    return [
        records_by_numeric_label[label]
        for label in sorted(records_by_numeric_label)
    ]


def validate_records(
    records: list[dict[str, Any]],
) -> None:
    """
    Validate the extracted Stanford breed metadata.
    """

    if len(records) != 120:
        raise ValueError(
            "Expected 120 Stanford Dogs breeds, "
            f"but extracted {len(records)}."
        )

    breed_ids = {
        record["breed_id"]
        for record in records
    }

    imagenet_ids = {
        record["imagenet_id"]
        for record in records
    }

    numeric_labels = {
        record["stanford_label"]
        for record in records
    }

    if len(breed_ids) != 120:
        raise ValueError(
            "Canonical breed IDs are not unique."
        )

    if len(imagenet_ids) != 120:
        raise ValueError(
            "ImageNet IDs are not unique."
        )

    if numeric_labels != set(range(1, 121)):
        raise ValueError(
            "Stanford numeric labels are not exactly 1 through 120."
        )

    total_images = sum(
        record["image_count"]
        for record in records
    )

    if total_images != 20580:
        raise ValueError(
            "Expected 20,580 Stanford Dogs images, "
            f"but counted {total_images}."
        )


def save_records(
    records: list[dict[str, Any]],
) -> None:
    """
    Write extracted breed metadata to JSON.
    """

    output = {
        "dataset_id": "stanford_dogs",
        "source": (
            "https://vision.stanford.edu/"
            "aditya86/ImageNetDogs/"
        ),
        "breed_count": len(records),
        "image_count": sum(
            record["image_count"]
            for record in records
        ),
        "generated_from": "file_list.mat",
        "breeds": records,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


def main() -> int:
    """
    Extract and validate Stanford Dogs breed labels.
    """

    print(
        "Extracting Stanford Dogs breed metadata..."
    )

    records = extract_breed_records()

    validate_records(records)

    save_records(records)

    print()
    print("Stanford Dogs metadata extraction complete.")
    print(f"Breeds: {len(records)}")

    total_images = sum(
        record["image_count"]
        for record in records
    )

    print(f"Images: {total_images}")
    print(f"Output: {OUTPUT_PATH}")

    print()
    print("First 5 breeds:")

    for record in records[:5]:
        print(
            f"  {record['stanford_label']:>3}: "
            f"{record['canonical_name']} "
            f"({record['imagenet_id']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())