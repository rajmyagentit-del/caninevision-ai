from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TAXONOMY_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "breed_taxonomy.json"
)


def normalize_text(value: str) -> str:
    """
    Normalize human-readable text.

    Steps:
    1. Strip whitespace.
    2. Convert Unicode characters to a normalized form.
    3. Collapse repeated whitespace.
    """

    value = unicodedata.normalize("NFKC", value)
    value = value.strip()
    value = re.sub(r"\s+", " ", value)

    return value


def canonicalize_breed_id(value: str) -> str:
    """
    Convert a breed label into a canonical lowercase snake_case ID.

    Example:
        "Golden Retriever" -> "golden_retriever"
        "German-Shepherd"  -> "german_shepherd"
    """

    value = normalize_text(value).lower()

    value = value.replace("&", " and ")

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


def humanize_breed_name(value: str) -> str:
    """
    Convert a dataset label into a readable breed name.

    Example:
        "golden_retriever" -> "Golden Retriever"
    """

    value = normalize_text(value)
    value = value.replace("_", " ")
    value = value.replace("-", " ")

    value = re.sub(r"\s+", " ", value)

    return value.title()


def build_breed_record(
    source_label: str,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Create a canonical breed record from a source dataset label.
    """

    canonical_id = canonicalize_breed_id(source_label)
    canonical_name = humanize_breed_name(source_label)

    return {
        "breed_id": canonical_id,
        "canonical_name": canonical_name,
        "aliases": [
            source_label
        ],
        "dataset_labels": {
            dataset_id: [
                source_label
            ]
        },
        "origin_country": None,
        "origin_region": None,
        "breed_group": None,
        "supported_by_model": False,
        "knowledge_available": False,
        "sources": []
    }


def merge_breed_record(
    existing_record: dict[str, Any],
    source_label: str,
    dataset_id: str,
) -> None:
    """
    Add a new dataset label to an existing canonical breed record.
    """

    if source_label not in existing_record["aliases"]:
        existing_record["aliases"].append(source_label)

    dataset_labels = existing_record["dataset_labels"]

    dataset_labels.setdefault(
        dataset_id,
        [],
    )

    if source_label not in dataset_labels[dataset_id]:
        dataset_labels[dataset_id].append(source_label)


def build_taxonomy_from_labels(
    labels: list[str],
    dataset_id: str,
) -> list[dict[str, Any]]:
    """
    Convert raw breed labels into unique canonical breed records.
    """

    records_by_id: dict[str, dict[str, Any]] = {}

    for label in labels:

        if not isinstance(label, str):
            raise TypeError(
                f"Breed label must be a string, got {type(label).__name__}"
            )

        cleaned_label = normalize_text(label)

        if not cleaned_label:
            continue

        canonical_id = canonicalize_breed_id(cleaned_label)

        if not canonical_id:
            continue

        if canonical_id not in records_by_id:
            records_by_id[canonical_id] = build_breed_record(
                source_label=cleaned_label,
                dataset_id=dataset_id,
            )

        else:
            merge_breed_record(
                existing_record=records_by_id[canonical_id],
                source_label=cleaned_label,
                dataset_id=dataset_id,
            )

    return sorted(
        records_by_id.values(),
        key=lambda record: record["breed_id"],
    )


def load_taxonomy() -> dict[str, Any]:
    """
    Load the existing breed taxonomy JSON file.
    """

    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"Missing taxonomy file: {TAXONOMY_PATH}"
        )

    with TAXONOMY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        taxonomy = json.load(file)

    if not isinstance(taxonomy, dict):
        raise TypeError(
            "breed_taxonomy.json must contain a top-level JSON object."
        )

    return taxonomy


def save_taxonomy(
    taxonomy: dict[str, Any],
) -> None:
    """
    Save the taxonomy JSON file with readable formatting.
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


def demo_labels() -> list[str]:
    """
    Temporary demonstration labels used to test normalization.

    These are not yet the production dataset labels.
    """

    return [
        "Golden Retriever",
        "golden_retriever",
        "German Shepherd",
        "German-Shepherd",
        "Belgian Malinois",
        "Siberian Husky",
        "Labrador Retriever",
        "labrador_retriever",
    ]


def main() -> int:
    """
    Run a temporary demonstration of the taxonomy builder.

    The demonstration proves the normalization logic works
    before we connect the script to real dataset labels.
    """

    taxonomy = load_taxonomy()

    labels = demo_labels()

    breed_records = build_taxonomy_from_labels(
        labels=labels,
        dataset_id="demo_dataset",
    )

    print("CanineVision AI breed taxonomy builder")
    print("--------------------------------------")

    for record in breed_records:
        print(
            f"{record['breed_id']:<25} "
            f"-> {record['canonical_name']}"
        )

    print()
    print(
        f"Input labels:       {len(labels)}"
    )

    print(
        f"Canonical breeds:   {len(breed_records)}"
    )

    print()
    print(
        "Demo completed successfully."
    )

    # Important:
    # We intentionally do NOT save these demo breeds
    # into breed_taxonomy.json yet.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())