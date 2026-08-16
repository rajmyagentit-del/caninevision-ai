from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW

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
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
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
    Freeze the early ResNet layers and unfreeze only:

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
) -> Path:

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

    if is_best:
        best_path = (
            FINETUNE_CHECKPOINT_DIR
            / "resnet50_finetune_best.pt"
        )

        torch.save(
            payload,
            best_path,
        )

    return epoch_path


def save_history(
    history: list[dict[str, Any]],
) -> None:

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

    args = parse_args()

    device = choose_device()

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

        checkpoint_path = (
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_accuracy=(
                    validation_accuracy
                ),
                is_best=is_best,
            )
        )

        print(
            "Checkpoint:",
            checkpoint_path
        )

        if is_best:
            print(
                "New best fine-tuned model."
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
            }
        )

    save_history(
        history
    )

    print()
    print(
        "Fine-tuning completed."
    )

    print(
        "Best validation accuracy:",
        f"{best_validation_accuracy:.4f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())