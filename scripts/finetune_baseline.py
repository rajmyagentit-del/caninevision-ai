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

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "resnet50_best_frozen.pt"
)

FINETUNE_CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "finetune"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "models"
    / "finetune_history.json"
)


def choose_device() -> torch.device:
    """
    Use CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load the frozen-backbone baseline checkpoint.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def configure_finetuning(
    model: nn.Module,
) -> None:
    """
    Freeze early ResNet layers and unfreeze:

    - layer4
    - final classifier head
    """

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count parameters updated during fine-tuning.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Fine-tune the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):

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

        if batch_index % 25 == 0:
            print(
                f"  train batch {batch_index}: "
                f"loss={loss.item():.4f}"
            )

    if total_examples == 0:
        raise RuntimeError(
            "Fine-tuning processed zero examples."
        )

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate without updating model weights.
    """

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:

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
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": float(total_examples),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_accuracy: float,
    is_best: bool,
) -> tuple[Path, Path | None]:
    """
    Save the epoch checkpoint locally.

    When this is the best epoch, also save a
    stable local best-checkpoint filename.
    """

    FINETUNE_CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_path = (
        FINETUNE_CHECKPOINT_DIR
        / f"resnet50_finetune_epoch_{epoch}.pt"
    )

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "validation_accuracy": (
            validation_accuracy
        ),
    }

    torch.save(
        payload,
        epoch_path,
    )

    best_path: Path | None = None

    if is_best:
        best_path = (
            FINETUNE_CHECKPOINT_DIR
            / "resnet50_finetune_best.pt"
        )

        torch.save(
            payload,
            best_path,
        )

    return epoch_path, best_path


def save_history(
    history: list[dict[str, Any]],
) -> None:
    """
    Save fine-tuning history locally.
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
    Parse fine-tuning command-line options.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune the CanineVision AI "
            "ResNet-50 baseline."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    return parser.parse_args()


def main() -> int:
    """
    Fine-tune layer4 and the classifier head.

    Best checkpoints and training history are
    automatically persisted when persistent
    storage is available.
    """

    args = parse_args()

    device = choose_device()

    storage_status = (
        persistence_status()
    )

    print(
        "CanineVision AI ResNet-50 fine-tuning"
    )
    print(
        "-------------------------------------"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Checkpoint: {args.checkpoint}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Learning rate: {args.learning_rate}"
    )

    print(
        "Persistent storage:",
        storage_status,
    )

    train_loader = create_dataloader(
        split="train",
        batch_size=args.batch_size,
    )

    validation_loader = create_dataloader(
        split="validation",
        batch_size=args.batch_size,
    )

    model = build_resnet50_baseline(
        freeze_backbone=False,
    )

    model = model.to(
        device
    )

    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    configure_finetuning(
        model
    )

    trainable_parameters = (
        count_trainable_parameters(
            model
        )
    )

    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )

    if (
        "validation_accuracy"
        in checkpoint
    ):
        print(
            "Loaded checkpoint validation accuracy:",
            f"{checkpoint['validation_accuracy']:.4f}"
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
        weight_decay=1e-4,
    )

    best_validation_accuracy = -1.0
    best_epoch: int | None = None
    best_local_checkpoint: Path | None = None
    best_persistent_checkpoint: Path | None = None

    history: list[
        dict[str, Any]
    ] = []

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print(
            f"Fine-tune epoch "
            f"{epoch}/{args.epochs}"
        )

        start_time = time.time()

        train_metrics = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
            )
        )

        validation_metrics = (
            evaluate(
                model=model,
                dataloader=validation_loader,
                criterion=criterion,
                device=device,
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

        is_best = (
            validation_accuracy
            > best_validation_accuracy
        )

        if is_best:
            best_validation_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch

        print(
            "Train loss:",
            f"{train_metrics['loss']:.4f}"
        )

        print(
            "Train accuracy:",
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            "Validation loss:",
            f"{validation_metrics['loss']:.4f}"
        )

        print(
            "Validation accuracy:",
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Epoch time: {duration:.1f}s"
        )

        (
            checkpoint_path,
            local_best_path,
        ) = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_accuracy=(
                validation_accuracy
            ),
            is_best=is_best,
        )

        print(
            "Checkpoint:",
            checkpoint_path,
        )

        if is_best:
            print(
                "New best fine-tuned model."
            )

            if local_best_path is None:
                raise RuntimeError(
                    "Best checkpoint path "
                    "was not created."
                )

            best_local_checkpoint = (
                local_best_path
            )

            persistent_checkpoint = (
                persist_checkpoint(
                    source=local_best_path,
                    destination_name=(
                        "resnet50_finetune_best.pt"
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
                    "Best fine-tuned checkpoint "
                    "persisted:",
                    persistent_checkpoint,
                )

            else:
                print(
                    "Persistent storage unavailable; "
                    "best fine-tuned checkpoint "
                    "remains local only."
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

        # Save and persist history after every
        # epoch so partial runs are recoverable.
        save_history(
            history
        )

        persistent_history = (
            persist_metrics(
                source=HISTORY_PATH,
                destination_name=(
                    "finetune_training_history.json"
                ),
            )
        )

        if (
            persistent_history
            is not None
        ):
            print(
                "Fine-tuning history persisted:",
                persistent_history,
            )

    print()
    print(
        "Fine-tuning completed."
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