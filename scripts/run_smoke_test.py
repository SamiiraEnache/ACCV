import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from configs.dataset_configs import DATASET_CONFIGS
from src.dataset import MedMNISTDataset
from src.ensemble import load_model_from_checkpoint
import src.ensemble as ensemble_module
from src.models import freeze_backbone, unfreeze_all
from src.submission import build_submission_csv
from src.train import train_one_epoch, validate
from src.transforms import get_train_transform, get_val_transform


def make_smoke_npz():
    Path("data").mkdir(exist_ok=True)

    n_train = 50
    n_val = 10
    n_test = 20
    height = 28
    width = 28
    n_classes = 3
    rng = np.random.default_rng(42)

    np.savez(
        "data/smokemnist.npz",
        train_images=rng.integers(0, 256, size=(n_train, height, width, 3), dtype=np.uint8),
        train_labels=rng.integers(0, n_classes, size=(n_train, 1), dtype=np.int64),
        val_images=rng.integers(0, 256, size=(n_val, height, width, 3), dtype=np.uint8),
        val_labels=rng.integers(0, n_classes, size=(n_val, 1), dtype=np.int64),
        test_images=rng.integers(0, 256, size=(n_test, height, width, 3), dtype=np.uint8),
        test_labels=rng.integers(0, n_classes, size=(n_test, 1), dtype=np.int64),
    )


def make_dataloaders():
    data = np.load("data/smokemnist.npz")
    train_transform = get_train_transform(224, "light", True, "smokemnist")
    val_transform = get_val_transform(224, True)

    train_dataset = MedMNISTDataset(data["train_images"], data["train_labels"], transform=train_transform)
    val_dataset = MedMNISTDataset(data["val_images"], data["val_labels"], transform=val_transform)
    test_dataset = MedMNISTDataset(data["test_images"], data["test_labels"], transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    images, labels = next(iter(train_loader))
    assert tuple(images.shape[1:]) == (3, 224, 224)
    assert labels.dtype == torch.int64
    assert int(labels.min()) >= 0 and int(labels.max()) < 3

    return train_loader, val_loader, test_loader, images


def test_model_forward(images):
    model = timm.create_model("resnet18", pretrained=False, num_classes=3, in_chans=3)
    logits = model(images)
    assert tuple(logits.shape) == (images.shape[0], 3)

    freeze_backbone(model)
    assert any(param.requires_grad for name, param in model.named_parameters() if "fc" in name)
    assert not any(param.requires_grad for name, param in model.named_parameters() if "fc" not in name)

    unfreeze_all(model)
    assert all(param.requires_grad for param in model.parameters())
    return model


def train_and_validate(model, train_loader, val_loader):
    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = []
    for epoch in range(2):
        train_loss, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_f1, per_class_f1 = validate(model, val_loader, criterion, device)
        history.append((train_loss, train_f1, val_loss, val_f1, per_class_f1))
        print(
            f"epoch={epoch + 1} train_loss={train_loss:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
    return history


def save_and_load_checkpoint(model, images):
    ckpt_dir = Path("checkpoints/smoke")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "test_ckpt.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    ensemble_module.build_model = lambda model_name, n_classes: timm.create_model(
        "resnet18",
        pretrained=False,
        num_classes=n_classes,
        in_chans=3,
    )
    loaded_model = load_model_from_checkpoint(ckpt_path, "resnet18", 3, torch.device("cpu"))
    with torch.no_grad():
        loaded_logits = loaded_model(images)
    assert loaded_logits.shape[1] == 3


def test_submission_csv():
    rng = np.random.default_rng(123)
    predictions = {}

    for dataset_name, config in DATASET_CONFIGS.items():
        n_test = 4
        if config["is_rgb"]:
            test_images = rng.integers(0, 256, size=(n_test, 28, 28, 3), dtype=np.uint8)
        else:
            test_images = rng.integers(0, 256, size=(n_test, 28, 28), dtype=np.uint8)

        np.savez(
            f"data/{dataset_name}.npz",
            train_images=test_images,
            train_labels=np.zeros((n_test, 1), dtype=np.int64),
            val_images=test_images,
            val_labels=np.zeros((n_test, 1), dtype=np.int64),
            test_images=test_images,
        )
        predictions[dataset_name] = rng.integers(0, config["n_classes"], size=n_test, dtype=np.int64)

    output_path = Path("submissions/smoke_submission.csv")
    submission = build_submission_csv(predictions, output_path)

    assert list(submission.columns) == ["id", "id_image_in_task", "task_name", "label"]
    assert submission["id"].is_monotonic_increasing
    assert set(submission["task_name"]) == set(DATASET_CONFIGS.keys())
    for _, group in submission.groupby("task_name"):
        assert int(group["id_image_in_task"].iloc[0]) == 0

    saved_submission = pd.read_csv(output_path)
    assert len(saved_submission) == len(submission)
    output_path.unlink()


def main():
    make_smoke_npz()
    train_loader, val_loader, _, images = make_dataloaders()
    model = test_model_forward(images)
    train_and_validate(model, train_loader, val_loader)
    save_and_load_checkpoint(model, images)
    test_submission_csv()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
