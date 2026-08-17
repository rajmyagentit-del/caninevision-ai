from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

from app.core.persistence import (
    persist_checkpoint,
    persist_metrics,
    persistence_status,
)
from app.models.baseline import build_resnet50_baseline
from app.services.dataset import create_dataloader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "training_history.json"
)


def choose_device() -> torch.device:
    """
    Use CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def calculate_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Calculate top-1 classification accuracy
    for one batch.
    """

    predictions = torch.argmax(
        logits,
        dim=1,
    )

    correct = (
        predictions == labels
    ).sum().item()

    return correct / labels.size(0)


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        if (
            max_batches is not None
            and batch_index > max_batches
        ):
            break

        images = batch["image"].to(
            device
        )

        labels = batch["label"].to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

        if batch_index % 10 == 0:
            print(
                f"  train batch {batch_index}: "
                f"loss={loss.item():.4f}"
            )

    if total_examples == 0:
        raise RuntimeError(
            "Training processed zero examples."
        )

    return {
        "loss": (
            total_loss
            / total_examples
        ),
        "accuracy": (
            total_correct
            / total_examples
        ),
        "examples": float(
            total_examples
        ),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """
    Evaluate the model without updating weights.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

        if (
            max_batches is not None
            and batch_index > max_batches
        ):
            break

        images = batch["image"].to(
            device
        )

        labels = batch["label"].to(
            device
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_examples += batch_size

    if total_examples == 0:
        raise RuntimeError(
            "Validation processed zero examples."
        )

    return {
        "loss": (
            total_loss
            / total_examples
        ),
        "accuracy": (
            total_correct
            / total_examples
        ),
        "examples": float(
            total_examples
        ),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_accuracy: float,
) -> Path:
    """
    Save model and optimizer state locally.
    """

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        CHECKPOINT_DIR
        / f"resnet50_epoch_{epoch}.pt"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "validation_accuracy": (
                validation_accuracy
            ),
        },
        path,
    )

    return path


def save_history(
    history: list[
        dict[str, Any]
    ],
) -> None:
    """
    Save training metrics as JSON locally.
    """

    HISTORY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

        file.write("\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line training options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the CanineVision AI "
            "ResNet-50 baseline."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()


def main() -> int:
    """
    Train the first CanineVision AI baseline.

    The best checkpoint is automatically copied
    to persistent storage when available.
    """

    args = parse_args()

    device = choose_device()

    storage_status = (
        persistence_status()
    )

    print(
        "CanineVision AI baseline training"
    )
    print(
        "---------------------------------"
    )
    print(
        f"Device: {device}"
    )
    print(
        f"Epochs: {args.epochs}"
    )
    print(
        f"Batch size: {args.batch_size}"
    )
    print(
        f"Learning rate: "
        f"{args.learning_rate}"
    )
    print(
        "Persistent storage:",
        storage_status,
    )

    train_loader = create_dataloader(
        split="train",
        batch_size=args.batch_size,
    )

    validation_loader = (
        create_dataloader(
            split="validation",
            batch_size=args.batch_size,
        )
    )

    model = build_resnet50_baseline(
        freeze_backbone=True,
    )

    model = model.to(
        device
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=args.learning_rate,
    )

    history: list[
        dict[str, Any]
    ] = []

    best_validation_accuracy = -1.0
    best_epoch: int | None = None
    best_local_checkpoint: Path | None = None
    best_persistent_checkpoint: Path | None = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        start_time = time.time()

        train_metrics = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                max_batches=(
                    args.max_train_batches
                ),
            )
        )

        validation_metrics = (
            evaluate(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
                max_batches=(
                    args.max_validation_batches
                ),
            )
        )

        duration = (
            time.time()
            - start_time
        )

        validation_accuracy = (
            validation_metrics[
                "accuracy"
            ]
        )

        print(
            "Train loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            "Train accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            "Validation loss: "
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            "Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Epoch time: {duration:.1f}s"
        )

        checkpoint_path = (
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_accuracy=(
                    validation_accuracy
                ),
            )
        )

        print(
            f"Checkpoint: "
            f"{checkpoint_path}"
        )

        is_best = (
            validation_accuracy
            > best_validation_accuracy
        )

        if is_best:
            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch
            best_local_checkpoint = (
                checkpoint_path
            )

            print(
                "New best baseline model."
            )

            persistent_checkpoint = (
                persist_checkpoint(
                    source=checkpoint_path,
                    destination_name=(
                        "resnet50_best_frozen.pt"
                    ),
                )
            )

            if (
                persistent_checkpoint
                is not None
            ):
                best_persistent_checkpoint = (
                    persistent_checkpoint
                )

                print(
                    "Best checkpoint persisted:",
                    persistent_checkpoint,
                )

            else:
                print(
                    "Persistent storage unavailable; "
                    "best checkpoint remains "
                    "local only."
                )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": (
                    validation_metrics
                ),
                "duration_seconds": (
                    duration
                ),
                "checkpoint": str(
                    checkpoint_path
                ),
                "is_best": is_best,
            }
        )

        # Save history after every epoch so that
        # metrics are not lost if training stops.
        save_history(
            history
        )

        persistent_history = (
            persist_metrics(
                source=HISTORY_PATH,
                destination_name=(
                    "baseline_training_history.json"
                ),
            )
        )

        if (
            persistent_history
            is not None
        ):
            print(
                "Training history persisted:",
                persistent_history,
            )

    print()
    print(
        "Training run completed."
    )

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.4f}"
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    if (
        best_local_checkpoint
        is not None
    ):
        print(
            "Best local checkpoint:",
            best_local_checkpoint,
        )

    if (
        best_persistent_checkpoint
        is not None
    ):
        print(
            "Best persistent checkpoint:",
            best_persistent_checkpoint,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())