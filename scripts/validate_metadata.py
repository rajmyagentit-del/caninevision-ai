from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "dataset_registry.json"
)

BREED_TAXONOMY_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "breed_taxonomy.json"
)


REQUIRED_DATASET_FIELDS = {
    "id",
    "name",
    "category",
    "role",
    "official_source",
    "license_status",
    "redistribution_status",
    "recommended_use",
    "status",
}


def load_json(path: Path) -> dict[str, Any]:
    """
    Load and parse a JSON file.

    Raises:
        FileNotFoundError:
            If the file does not exist.

        ValueError:
            If the file contains invalid JSON.

        TypeError:
            If the top-level JSON value is not an object.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: "
            f"line {error.lineno}, column {error.colno}"
        ) from error

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected top-level JSON object in {path}, "
            f"got {type(data).__name__}"
        )

    return data


def validate_dataset_registry(
    registry: dict[str, Any],
) -> list[str]:
    """
    Validate the dataset registry structure.

    Returns:
        A list of validation error messages.
        An empty list means validation passed.
    """

    errors: list[str] = []

    if registry.get("project") != "CanineVision AI":
        errors.append(
            "dataset_registry.json must contain "
            "'project': 'CanineVision AI'"
        )

    datasets = registry.get("datasets")

    if not isinstance(datasets, list):
        errors.append(
            "dataset_registry.json field 'datasets' must be a list."
        )
        return errors

    if not datasets:
        errors.append(
            "dataset_registry.json contains no datasets."
        )
        return errors

    seen_ids: set[str] = set()

    for index, dataset in enumerate(datasets, start=1):

        if not isinstance(dataset, dict):
            errors.append(
                f"Dataset entry #{index} must be a JSON object."
            )
            continue

        missing_fields = REQUIRED_DATASET_FIELDS - dataset.keys()

        if missing_fields:
            errors.append(
                f"Dataset entry #{index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        dataset_id = dataset.get("id")

        if isinstance(dataset_id, str):

            if dataset_id in seen_ids:
                errors.append(
                    f"Duplicate dataset id detected: {dataset_id}"
                )

            seen_ids.add(dataset_id)

        else:
            errors.append(
                f"Dataset entry #{index} has an invalid 'id'."
            )

        known_classes = dataset.get("known_classes")

        if (
            known_classes is not None
            and not isinstance(known_classes, int)
        ):
            errors.append(
                f"Dataset '{dataset_id}' has invalid "
                "'known_classes'; expected integer or null."
            )

        known_images = dataset.get("known_images")

        if (
            known_images is not None
            and not isinstance(known_images, int)
        ):
            errors.append(
                f"Dataset '{dataset_id}' has invalid "
                "'known_images'; expected integer or null."
            )

    return errors


def validate_breed_taxonomy(
    taxonomy: dict[str, Any],
) -> list[str]:
    """
    Validate the breed taxonomy schema.

    Returns:
        A list of validation errors.
    """

    errors: list[str] = []

    required_fields = {
        "project",
        "version",
        "normalization_policy",
        "special_classes",
        "breed_schema",
        "breeds",
    }

    missing_fields = required_fields - taxonomy.keys()

    if missing_fields:
        errors.append(
            "breed_taxonomy.json is missing fields: "
            f"{sorted(missing_fields)}"
        )

    if taxonomy.get("project") != "CanineVision AI":
        errors.append(
            "breed_taxonomy.json must contain "
            "'project': 'CanineVision AI'"
        )

    breeds = taxonomy.get("breeds")

    if not isinstance(breeds, list):
        errors.append(
            "breed_taxonomy.json field 'breeds' must be a list."
        )

    special_classes = taxonomy.get("special_classes")

    if not isinstance(special_classes, list):
        errors.append(
            "breed_taxonomy.json field "
            "'special_classes' must be a list."
        )

    return errors


def print_validation_summary(
    dataset_count: int,
    taxonomy_breed_count: int,
) -> None:
    """
    Print a concise success summary.
    """

    print("CanineVision AI metadata validation")
    print("-----------------------------------")
    print("Dataset registry: VALID")
    print("Breed taxonomy:   VALID")
    print(f"Registered datasets: {dataset_count}")
    print(f"Canonical breeds:    {taxonomy_breed_count}")
    print()
    print("Metadata validation completed successfully.")


def main() -> int:
    """
    Program entry point.

    Returns:
        0 when validation succeeds.
        1 when validation fails.
    """

    try:
        dataset_registry = load_json(DATASET_REGISTRY_PATH)
        breed_taxonomy = load_json(BREED_TAXONOMY_PATH)

        errors = []

        errors.extend(
            validate_dataset_registry(dataset_registry)
        )

        errors.extend(
            validate_breed_taxonomy(breed_taxonomy)
        )

        if errors:
            print(
                "Metadata validation FAILED:",
                file=sys.stderr,
            )

            for error in errors:
                print(
                    f" - {error}",
                    file=sys.stderr,
                )

            return 1

        dataset_count = len(
            dataset_registry["datasets"]
        )

        taxonomy_breed_count = len(
            breed_taxonomy["breeds"]
        )

        print_validation_summary(
            dataset_count=dataset_count,
            taxonomy_breed_count=taxonomy_breed_count,
        )

        return 0

    except (
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as error:

        print(
            f"Metadata validation FAILED: {error}",
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())