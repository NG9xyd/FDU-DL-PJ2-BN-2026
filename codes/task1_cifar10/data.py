from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    classes: tuple[str, ...]


def _transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[object] = []
    if augment:
        train_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return transforms.Compose(train_steps), eval_transform


def _limit_dataset(dataset: Dataset, n_items: int, seed: int) -> Dataset:
    if n_items <= 0 or n_items >= len(dataset):
        return dataset
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    return Subset(dataset, indices[:n_items].tolist())


def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    val_size: int = 5000,
    seed: int = 42,
    augment: bool = True,
    download: bool = True,
    fake_data: bool = False,
    train_items: int = -1,
    test_items: int = -1,
) -> DataLoaders:
    train_transform, eval_transform = _transforms(augment)

    if fake_data:
        full_train_aug = datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
            random_offset=0,
        )
        full_train_eval = datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_transform,
            random_offset=0,
        )
        test_dataset: Dataset = datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_transform,
            random_offset=1000,
        )
        val_size = min(val_size, 128)
    else:
        root = str(Path(data_dir))
        full_train_aug = datasets.CIFAR10(
            root=root, train=True, download=download, transform=train_transform
        )
        full_train_eval = datasets.CIFAR10(
            root=root, train=True, download=False, transform=eval_transform
        )
        test_dataset = datasets.CIFAR10(
            root=root, train=False, download=download, transform=eval_transform
        )

    if not 0 < val_size < len(full_train_aug):
        raise ValueError(f"val_size must be in [1, {len(full_train_aug) - 1}]")

    generator = torch.Generator().manual_seed(seed)
    train_size = len(full_train_aug) - val_size
    train_subset, val_subset_aug = random_split(
        full_train_aug, [train_size, val_size], generator=generator
    )
    val_dataset = Subset(full_train_eval, val_subset_aug.indices)
    train_dataset = _limit_dataset(train_subset, train_items, seed)
    test_dataset = _limit_dataset(test_dataset, test_items, seed + 1)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return DataLoaders(train_loader, val_loader, test_loader, CIFAR10_CLASSES)
