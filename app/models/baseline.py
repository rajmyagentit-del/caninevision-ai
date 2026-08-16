from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


NUM_CLASSES = 120


def build_resnet50_baseline(
    num_classes: int = NUM_CLASSES,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Build a ResNet-50 transfer-learning baseline.

    Args:
        num_classes:
            Number of output dog-breed classes.

        freeze_backbone:
            If True, freeze the pretrained convolutional
            feature extractor and train only the final
            classification layer initially.

    Returns:
        Configured PyTorch model.
    """

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(
        weights=weights,
    )

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    input_features = model.fc.in_features

    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    return model


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Return total and trainable parameter counts.
    """

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total, trainable


def top_k_predictions(
    logits: torch.Tensor,
    k: int = 5,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Convert raw model logits into top-k probabilities
    and class indices.
    """

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=k,
            dim=1,
        )
    )

    return (
        top_probabilities,
        top_indices,
    )


def run_shape_test() -> None:
    """
    Run a lightweight forward-pass test using
    synthetic input.

    This verifies that the model accepts a batch
    shaped like our real image tensors and returns
    120 output logits per image.
    """

    model = build_resnet50_baseline(
        freeze_backbone=True,
    )

    model.eval()

    fake_batch = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.no_grad():
        logits = model(
            fake_batch
        )

    total_parameters, trainable_parameters = (
        count_parameters(
            model
        )
    )

    print(
        "CanineVision AI ResNet-50 baseline"
    )
    print(
        "----------------------------------"
    )
    print(
        "Input shape: ",
        tuple(fake_batch.shape),
    )
    print(
        "Output shape:",
        tuple(logits.shape),
    )
    print(
        "Total parameters:",
        f"{total_parameters:,}",
    )
    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}",
    )

    top_probabilities, top_indices = (
        top_k_predictions(
            logits,
            k=5,
        )
    )

    print(
        "Top-k probability shape:",
        tuple(
            top_probabilities.shape
        ),
    )

    print(
        "Top-k index shape:",
        tuple(
            top_indices.shape
        ),
    )


if __name__ == "__main__":
    run_shape_test()