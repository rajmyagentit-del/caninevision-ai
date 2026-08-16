from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANFORD_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "stanford_dogs"
)

IMAGES_DIR = STANFORD_ROOT / "Images"

LABELS_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "stanford_dogs_labels.json"
)

TAXONOMY_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "breed_taxonomy.json"
)


SUPPORTED_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file and verify that the top-level
    JSON value is an object.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object in {path}, "
            f"got {type(data).__name__}"
        )

    return data


def get_actual_breed_directories() -> set[str]:
    """
    Return all breed directory names found under Images/.
    """

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Stanford Images directory not found: {IMAGES_DIR}"
        )

    return {
        path.name
        for path in IMAGES_DIR.iterdir()
        if path.is_dir()
    }


def get_expected_breed_directories(
    labels_data: dict[str, Any],
) -> set[str]:
    """
    Return the expected breed directories from
    stanford_dogs_labels.json.
    """

    breeds = labels_data.get("breeds")

    if not isinstance(breeds, list):
        raise TypeError(
            "stanford_dogs_labels.json field "
            "'breeds' must be a list."
        )

    expected = set()

    for record in breeds:

        if not isinstance(record, dict):
            raise TypeError(
                "Every Stanford breed record "
                "must be a JSON object."
            )

        source_directory = record.get(
            "source_directory"
        )

        if not isinstance(source_directory, str):
            raise TypeError(
                "Every Stanford breed record must "
                "contain a string 'source_directory'."
            )

        expected.add(source_directory)

    return expected


def count_image_files() -> int:
    """
    Count supported image files under all Stanford
    breed directories.
    """

    count = 0

    for path in IMAGES_DIR.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_IMAGE_SUFFIXES
        ):
            count += 1

    return count


def validate_directory_mapping(
    expected: set[str],
    actual: set[str],
) -> list[str]:
    """
    Compare expected and actual breed directories.
    """

    errors: list[str] = []

    missing = sorted(
        expected - actual
    )

    unexpected = sorted(
        actual - expected
    )

    if missing:
        errors.append(
            "Missing breed directories: "
            + ", ".join(missing)
        )

    if unexpected:
        errors.append(
            "Unexpected breed directories: "
            + ", ".join(unexpected)
        )

    return errors


def validate_taxonomy(
    taxonomy: dict[str, Any],
    labels_data: dict[str, Any],
) -> list[str]:
    """
    Verify that every Stanford breed is represented
    in the canonical taxonomy.
    """

    errors: list[str] = []

    taxonomy_breeds = taxonomy.get("breeds")

    if not isinstance(taxonomy_breeds, list):
        errors.append(
            "breed_taxonomy.json field "
            "'breeds' must be a list."
        )

        return errors

    taxonomy_ids = {
        record.get("breed_id")
        for record in taxonomy_breeds
        if isinstance(record, dict)
    }

    stanford_ids = {
        record.get("breed_id")
        for record in labels_data["breeds"]
        if isinstance(record, dict)
    }

    missing_taxonomy_ids = sorted(
        breed_id
        for breed_id in (
            stanford_ids - taxonomy_ids
        )
        if breed_id is not None
    )

    if missing_taxonomy_ids:
        errors.append(
            "Stanford breeds missing from taxonomy: "
            + ", ".join(missing_taxonomy_ids)
        )

    return errors


def validate_image_counts(
    labels_data: dict[str, Any],
) -> list[str]:
    """
    Verify the total extracted image count and
    per-breed image counts.
    """

    errors: list[str] = []

    expected_total = labels_data.get(
        "image_count"
    )

    if not isinstance(expected_total, int):
        errors.append(
            "stanford_dogs_labels.json field "
            "'image_count' must be an integer."
        )

        return errors

    actual_total = count_image_files()

    if actual_total != expected_total:
        errors.append(
            f"Expected {expected_total} image files, "
            f"found {actual_total}."
        )

    for record in labels_data["breeds"]:

        source_directory = record[
            "source_directory"
        ]

        expected_count = record[
            "image_count"
        ]

        directory = (
            IMAGES_DIR
            / source_directory
        )

        actual_count = sum(
            1
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_IMAGE_SUFFIXES
            )
        )

        if actual_count != expected_count:
            errors.append(
                f"{source_directory}: "
                f"expected {expected_count} images, "
                f"found {actual_count}."
            )

    return errors


def main() -> int:
    """
    Validate the local Stanford Dogs dataset.
    """

    print(
        "CanineVision AI Stanford Dogs "
        "dataset validation"
    )

    print(
        "-----------------------------------------"
    )

    labels_data = load_json(
        LABELS_PATH
    )

    taxonomy = load_json(
        TAXONOMY_PATH
    )

    expected_directories = (
        get_expected_breed_directories(
            labels_data
        )
    )

    actual_directories = (
        get_actual_breed_directories()
    )

    errors: list[str] = []

    errors.extend(
        validate_directory_mapping(
            expected=expected_directories,
            actual=actual_directories,
        )
    )

    errors.extend(
        validate_taxonomy(
            taxonomy=taxonomy,
            labels_data=labels_data,
        )
    )

    errors.extend(
        validate_image_counts(
            labels_data=labels_data,
        )
    )

    print(
        f"Expected breed directories: "
        f"{len(expected_directories)}"
    )

    print(
        f"Actual breed directories:   "
        f"{len(actual_directories)}"
    )

    print(
        f"Image files found:          "
        f"{count_image_files()}"
    )

    print(
        f"Canonical taxonomy breeds:  "
        f"{len(taxonomy['breeds'])}"
    )

    if errors:
        print()
        print(
            "Dataset validation FAILED:"
        )

        for error in errors:
            print(
                f" - {error}"
            )

        return 1

    print()
    print(
        "Stanford Dogs dataset validation PASSED."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())