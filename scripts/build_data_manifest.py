from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANFORD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stanford_dogs"
)

TRAIN_LIST_PATH = STANFORD_ROOT / "train_list.mat"
TEST_LIST_PATH = STANFORD_ROOT / "test_list.mat"

TAXONOMY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "breed_taxonomy.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stanford_data_manifest.json"
)

VALIDATION_FRACTION = 0.15
RANDOM_SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file and require a top-level object.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object in {path}."
        )

    return data


def extract_matlab_string(value: Any) -> str:
    """
    Convert Stanford's nested MATLAB/NumPy string
    representation into a normal Python string.
    """

    current = value

    while hasattr(current, "ndim") and current.ndim > 0:
        if current.size != 1:
            raise ValueError(
                "Expected a single MATLAB string value."
            )

        current = current.item()

    if not isinstance(current, str):
        current = str(current)

    return current.strip()


def build_label_mapping(
    taxonomy: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """
    Map Stanford numeric labels to canonical
    CanineVision AI breed metadata.
    """

    breeds = taxonomy.get("breeds")

    if not isinstance(breeds, list):
        raise TypeError(
            "breed_taxonomy.json field 'breeds' "
            "must be a list."
        )

    mapping: dict[int, dict[str, Any]] = {}

    for breed in breeds:
        if not isinstance(breed, dict):
            continue

        stanford = (
            breed
            .get("dataset_labels", {})
            .get("stanford_dogs")
        )

        if not isinstance(stanford, dict):
            continue

        numeric_label = stanford.get(
            "numeric_label"
        )

        if not isinstance(numeric_label, int):
            continue

        mapping[numeric_label] = {
            "breed_id": breed["breed_id"],
            "canonical_name": breed["canonical_name"],
            "class_index": numeric_label - 1,
        }

    if len(mapping) != 120:
        raise ValueError(
            f"Expected 120 Stanford label mappings, "
            f"found {len(mapping)}."
        )

    return mapping


def load_mat_split(
    path: Path,
    split_name: str,
    label_mapping: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Read one Stanford .mat split and convert it
    into normalized manifest records.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stanford split file: {path}"
        )

    data = loadmat(path)

    file_list = data["file_list"]
    labels = data["labels"]

    if len(file_list) != len(labels):
        raise ValueError(
            f"{split_name}: file count and label count differ."
        )

    records: list[dict[str, Any]] = []

    for file_entry, label_entry in zip(
        file_list,
        labels,
        strict=True,
    ):
        relative_path = extract_matlab_string(
            file_entry
        )

        numeric_label = int(
            label_entry.flat[0]
        )

        breed = label_mapping.get(
            numeric_label
        )

        if breed is None:
            raise ValueError(
                f"Unknown Stanford numeric label: "
                f"{numeric_label}"
            )

        full_path = (
            STANFORD_ROOT
            / "Images"
            / relative_path
        )

        if not full_path.exists():
            raise FileNotFoundError(
                f"Image referenced by Stanford metadata "
                f"does not exist: {full_path}"
            )

        records.append(
            {
                "image_path": relative_path,
                "breed_id": breed["breed_id"],
                "canonical_name": (
                    breed["canonical_name"]
                ),
                "stanford_label": numeric_label,
                "class_index": (
                    breed["class_index"]
                ),
                "split": split_name,
            }
        )

    return records


def create_train_validation_split(
    training_records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Create a deterministic, class-stratified
    validation set from Stanford's official
    training split.

    Stanford's official test set remains untouched.
    """

    by_class: dict[
        int,
        list[dict[str, Any]]
    ] = defaultdict(list)

    for record in training_records:
        by_class[
            record["class_index"]
        ].append(record)

    rng = random.Random(
        RANDOM_SEED
    )

    final_train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []

    for class_index in sorted(by_class):
        class_records = list(
            by_class[class_index]
        )

        rng.shuffle(class_records)

        validation_count = max(
            1,
            round(
                len(class_records)
                * VALIDATION_FRACTION
            ),
        )

        validation_records = (
            class_records[
                :validation_count
            ]
        )

        train_records = (
            class_records[
                validation_count:
            ]
        )

        for record in validation_records:
            record = dict(record)
            record["split"] = "validation"
            validation.append(record)

        for record in train_records:
            record = dict(record)
            record["split"] = "train"
            final_train.append(record)

    return final_train, validation


def validate_manifest(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    """
    Validate split sizes, labels, and leakage.
    """

    all_records = (
        train_records
        + validation_records
        + test_records
    )

    if len(all_records) != 20580:
        raise ValueError(
            "Expected 20,580 total records, "
            f"found {len(all_records)}."
        )

    train_paths = {
        record["image_path"]
        for record in train_records
    }

    validation_paths = {
        record["image_path"]
        for record in validation_records
    }

    test_paths = {
        record["image_path"]
        for record in test_records
    }

    if train_paths & validation_paths:
        raise ValueError(
            "Train/validation leakage detected."
        )

    if train_paths & test_paths:
        raise ValueError(
            "Train/test leakage detected."
        )

    if validation_paths & test_paths:
        raise ValueError(
            "Validation/test leakage detected."
        )

    class_indices = {
        record["class_index"]
        for record in all_records
    }

    if class_indices != set(range(120)):
        raise ValueError(
            "Expected class indices 0 through 119."
        )


def save_manifest(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:
    """
    Save the complete model-ready data manifest.
    """

    output = {
        "dataset_id": "stanford_dogs",
        "random_seed": RANDOM_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "class_count": 120,
        "splits": {
            "train": len(train_records),
            "validation": len(
                validation_records
            ),
            "test": len(test_records),
        },
        "records": (
            train_records
            + validation_records
            + test_records
        ),
    }

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
    Build a reproducible model-ready Stanford
    Dogs data manifest.
    """

    print(
        "Building Stanford Dogs data manifest..."
    )

    taxonomy = load_json(
        TAXONOMY_PATH
    )

    label_mapping = build_label_mapping(
        taxonomy
    )

    official_train = load_mat_split(
        path=TRAIN_LIST_PATH,
        split_name="official_train",
        label_mapping=label_mapping,
    )

    official_test = load_mat_split(
        path=TEST_LIST_PATH,
        split_name="test",
        label_mapping=label_mapping,
    )

    train_records, validation_records = (
        create_train_validation_split(
            official_train
        )
    )

    validate_manifest(
        train_records=train_records,
        validation_records=validation_records,
        test_records=official_test,
    )

    save_manifest(
        train_records=train_records,
        validation_records=validation_records,
        test_records=official_test,
    )

    print()
    print("Data manifest created successfully.")
    print(f"Train:      {len(train_records)}")
    print(
        f"Validation: {len(validation_records)}"
    )
    print(f"Test:       {len(official_test)}")
    print(
        "Total:      "
        f"{len(train_records) + len(validation_records) + len(official_test)}"
    )
    print()
    print(f"Output: {OUTPUT_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())