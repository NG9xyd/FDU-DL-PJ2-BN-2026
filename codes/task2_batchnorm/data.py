from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _limit(dataset: Dataset, number: int, seed: int) -> Dataset:
    if number <= 0 or number >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:number].tolist()
    return Subset(dataset, indices)


def build_loaders(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    val_size: int,
    seed: int,
    train_items: int = -1,
    test_items: int = -1,
    augment: bool = False,
    fake_data: bool = False,
) -> Loaders:
    base_steps: list[object] = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    train_steps: list[object] = []
    if augment:
        train_steps.extend(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        )
    train_transform = transforms.Compose(train_steps + base_steps)
    eval_transform = transforms.Compose(base_steps)

    if fake_data:
        train_aug = datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_transform,
            random_offset=0,
        )
        train_eval = datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_transform,
            random_offset=0,
        )
        test: Dataset = datasets.FakeData(
            size=128,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_transform,
            random_offset=1000,
        )
        val_size = min(val_size, 128)
    else:
        root = str(data_dir)
        train_aug = datasets.CIFAR10(
            root=root, train=True, download=False, transform=train_transform
        )
        train_eval = datasets.CIFAR10(
            root=root, train=True, download=False, transform=eval_transform
        )
        test = datasets.CIFAR10(
            root=root, train=False, download=False, transform=eval_transform
        )

    if not 0 < val_size < len(train_aug):
        raise ValueError(f"val_size must be between 1 and {len(train_aug) - 1}")
    train_size = len(train_aug) - val_size
    train_subset, val_subset = random_split(
        train_aug,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_subset = _limit(train_subset, train_items, seed)
    validation = Subset(train_eval, val_subset.indices)
    test = _limit(test, test_items, seed + 1)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return Loaders(
        train=DataLoader(
            train_subset,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            **common,
        ),
        val=DataLoader(validation, shuffle=False, **common),
        test=DataLoader(test, shuffle=False, **common),
    )
