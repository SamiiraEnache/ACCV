import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from configs.base_config import BASE_CONFIG


class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = np.asarray(images)
        self.labels = np.asarray(labels).squeeze().astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.asarray(self.images[index])

        if image.ndim == 2:
            image = Image.fromarray(image).convert("L")
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = Image.fromarray(image.squeeze(-1)).convert("L")
        else:
            image = Image.fromarray(image).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, int(self.labels[index])


def get_dataloaders(
    dataset_name: str,
    image_size: int,
    batch_size: int,
    train_transform,
    val_transform,
    num_workers: int = 2,
    use_trainval: bool = False,
):
    data_root = BASE_CONFIG["data_root"]
    npz_path = os.path.join(data_root, f"{dataset_name}.npz")

    with np.load(npz_path) as data:
        train_images = data["train_images"]
        train_labels = data["train_labels"]
        val_images = data["val_images"]
        val_labels = data["val_labels"]
        test_images = data["test_images"]
        test_labels = data["test_labels"] if "test_labels" in data.files else np.zeros(len(test_images), dtype=np.int64)

    if use_trainval:
        train_images = np.concatenate([train_images, val_images], axis=0)
        train_labels = np.concatenate([train_labels, val_labels], axis=0)

    train_dataset = MedMNISTDataset(train_images, train_labels, transform=train_transform)
    val_dataset = MedMNISTDataset(val_images, val_labels, transform=val_transform)
    test_dataset = MedMNISTDataset(test_images, test_labels, transform=val_transform)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader
