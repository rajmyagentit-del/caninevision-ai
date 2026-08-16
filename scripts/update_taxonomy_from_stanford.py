from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANFORD_LABELS_PATH = (
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


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file and verify the top-level value is an object.
    """

    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, "
            f"got {type(data).__name__}"
        )

    return data


def build_taxonomy_record(
    stanford_record: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one Stanford Dogs breed record into the
    CanineVision AI canonical taxonomy schema.
    """

    return {
        "breed_id": stanford_record["breed_id"],
        "canonical_name": stanford_record["canonical_name"],
        "aliases": [
            stanford_record["canonical_name"]
        ],
        "dataset_labels": {
            "stanford_dogs": {
                "numeric_label": stanford_record["stanford_label"],
                "source_directory": stanford_record["source_directory"],
                "imagenet_id": stanford_record["imagenet_id"]
            }
        },
        "origin_country": None,
        "origin_region": None,
        "breed_group": None,
        "supported_by_model": True,
        "knowledge_available": False,
        "sources": [
            {
                "type": "dataset",
                "dataset_id": "stanford_dogs",
                "source": (
                    "https://vision.stanford.edu/"
                    "aditya86/ImageNetDogs/"
                )
            }
        ]
    }


def validate_stanford_input(
    stanford_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Validate the extracted Stanford Dogs label file.
    """

    if stanford_data.get("dataset_id") != "stanford_dogs":
        raise ValueError(
            "Expected dataset_id 'stanford_dogs'."
        )

    if stanford_data.get("breed_count") != 120:
        raise ValueError(
            "Expected Stanford Dogs breed_count to equal 120."
        )

    if stanford_data.get("image_count") != 20580:
        raise ValueError(
            "Expected Stanford Dogs image_count to equal 20580."
        )

    breeds = stanford_data.get("breeds")

    if not isinstance(breeds, list):
        raise TypeError(
            "Stanford Dogs field 'breeds' must be a list."
        )

    if len(breeds) != 120:
        raise ValueError(
            f"Expected 120 breeds, found {len(breeds)}."
        )

    return breeds


def build_taxonomy_records(
    stanford_breeds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build and validate 120 canonical taxonomy records.
    """

    records = [
        build_taxonomy_record(record)
        for record in stanford_breeds
    ]

    breed_ids = {
        record["breed_id"]
        for record in records
    }

    if len(breed_ids) != 120:
        raise ValueError(
            "Expected 120 unique canonical breed IDs."
        )

    numeric_labels = {
        record["dataset_labels"]["stanford_dogs"]["numeric_label"]
        for record in records
    }

    if numeric_labels != set(range(1, 121)):
        raise ValueError(
            "Stanford numeric labels must be exactly 1 through 120."
        )

    return sorted(
        records,
        key=lambda record: (
            record["dataset_labels"]
            ["stanford_dogs"]
            ["numeric_label"]
        ),
    )


def update_taxonomy(
    taxonomy: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Replace the current breed list with the validated
    Stanford Dogs canonical taxonomy records.
    """

    taxonomy["breeds"] = records

    return taxonomy


def save_taxonomy(
    taxonomy: dict[str, Any],
) -> None:
    """
    Save the updated canonical taxonomy.
    """

    with TAXONOMY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            taxonomy,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


def main() -> int:
    """
    Build the initial production breed taxonomy
    from Stanford Dogs metadata.
    """

    print(
        "Building CanineVision AI taxonomy "
        "from Stanford Dogs..."
    )

    stanford_data = load_json(
        STANFORD_LABELS_PATH
    )

    taxonomy = load_json(
        TAXONOMY_PATH
    )

    stanford_breeds = validate_stanford_input(
        stanford_data
    )

    records = build_taxonomy_records(
        stanford_breeds
    )

    updated_taxonomy = update_taxonomy(
        taxonomy=taxonomy,
        records=records,
    )

    save_taxonomy(
        updated_taxonomy
    )

    print()
    print("Breed taxonomy update complete.")
    print(f"Canonical breeds: {len(records)}")
    print(f"Output: {TAXONOMY_PATH}")

    print()
    print("First 5 canonical breeds:")

    for record in records[:5]:
        stanford_metadata = (
            record["dataset_labels"]["stanford_dogs"]
        )

        print(
            f"  {stanford_metadata['numeric_label']:>3}: "
            f"{record['canonical_name']} "
            f"-> {record['breed_id']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())