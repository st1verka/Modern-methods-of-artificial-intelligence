"""Reproducible Transfer Learning experiment for Lab 1.

The script uses three MNIST image classes (digits 0, 1 and 2),
a MobileNetV2 backbone pretrained on ImageNet, and compares feature extraction
against fine-tuning of the last backbone block.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms


CLASS_IDS = (0, 1, 2)
CLASS_NAMES = ("digit_0", "digit_1", "digit_2")


@dataclass
class Result:
    strategy: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    trainable_parameters: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_classes(dataset: Dataset, targets: list[int], per_class: int, seed: int) -> Subset:
    """Return a balanced, deterministic subset of the chosen three classes."""
    rng = random.Random(seed)
    indices_by_class = {class_id: [] for class_id in CLASS_IDS}
    for index, target in enumerate(targets):
        if target in indices_by_class:
            indices_by_class[target].append(index)
    selected = []
    for class_id in CLASS_IDS:
        candidates = indices_by_class[class_id]
        rng.shuffle(candidates)
        if len(candidates) < per_class:
            raise ValueError(f"Not enough examples of class {class_id}")
        selected.extend(candidates[:per_class])
    rng.shuffle(selected)
    return Subset(dataset, selected)


class RelabeledSubset(Dataset):
    """Subset wrapper that maps the selected original labels to 0..2."""

    def __init__(self, subset: Subset) -> None:
        self.subset = subset

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        return image, CLASS_IDS.index(label)


def make_loaders(data_dir: Path, train_per_class: int, test_per_class: int,
                 batch_size: int, seed: int) -> tuple[DataLoader, DataLoader]:
    weights = models.MobileNet_V2_Weights.DEFAULT
    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(weights.transforms().mean, weights.transforms().std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(weights.transforms().mean, weights.transforms().std),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.MNIST(data_dir, train=False, download=True, transform=test_transform)
    train_subset = select_classes(train_set, train_set.targets.tolist(), train_per_class, seed)
    test_subset = select_classes(test_set, test_set.targets.tolist(), test_per_class, seed)
    return (
        DataLoader(RelabeledSubset(train_subset), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(RelabeledSubset(test_subset), batch_size=batch_size, shuffle=False, num_workers=0),
    )


def build_model(strategy: str, device: torch.device) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    if strategy == "fine_tuning":
        # Unfreeze only the final inverted-residual block and terminal convolution.
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    return model.to(device)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[Result, np.ndarray]:
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            truth.extend(labels.tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth, predicted, average="macro", zero_division=0
    )
    result = Result(
        strategy="",
        accuracy=float(accuracy_score(truth, predicted)),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return result, confusion_matrix(truth, predicted, labels=range(len(CLASS_NAMES)))


def save_confusion_matrix(matrix: np.ndarray, strategy: str, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)),
             xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             ylabel="True class", xlabel="Predicted class",
             title=f"Confusion matrix: {strategy.replace('_', ' ')}")
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    threshold = matrix.max() / 2 if matrix.max() else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center",
                      color="white" if matrix[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_dir / f"confusion_matrix_{strategy}.png", dpi=180)
    plt.close(fig)


def train_strategy(strategy: str, train_loader: DataLoader, test_loader: DataLoader,
                   device: torch.device, epochs: int, learning_rate: float,
                   output_dir: Path) -> Result:
    model = build_model(strategy, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate, weight_decay=1e-4,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        print(f"{strategy}: epoch {epoch}/{epochs}, loss={total_loss / seen:.4f}")
    result, matrix = evaluate(model, test_loader, device)
    result.strategy = strategy
    save_confusion_matrix(matrix, strategy, output_dir)
    torch.save({"state_dict": model.state_dict(), "classes": CLASS_NAMES},
               output_dir / f"mobilenetv2_{strategy}.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--train-per-class", type=int, default=600)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--epochs-feature", type=int, default=3)
    parser.add_argument("--epochs-finetune", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader = make_loaders(args.data_dir, args.train_per_class,
                                             args.test_per_class, args.batch_size, args.seed)
    results = [
        train_strategy("feature_extraction", train_loader, test_loader, device,
                       args.epochs_feature, 1e-3, args.output_dir),
        train_strategy("fine_tuning", train_loader, test_loader, device,
                       args.epochs_finetune, 1e-4, args.output_dir),
    ]
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as stream:
        json.dump({"classes": CLASS_NAMES, "device": str(device), "seed": args.seed,
                   "train_per_class": args.train_per_class, "test_per_class": args.test_per_class,
                   "feature_extraction_lr": 1e-3, "fine_tuning_lr": 1e-4}, stream, indent=2)
    print("\nResults:")
    for result in results:
        print(asdict(result))


if __name__ == "__main__":
    main()
