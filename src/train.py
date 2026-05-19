import numpy as np
import torch
from sklearn.metrics import f1_score


def _unpack_batch(batch):
    images, labels = batch
    labels = labels.view(-1).long()
    return images, labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    for batch in loader:
        images, labels = _unpack_batch(batch)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_targets.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    average_loss = total_loss / max(total_samples, 1)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return average_loss, macro_f1


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []
    n_classes = None

    with torch.no_grad():
        for batch in loader:
            images, labels = _unpack_batch(batch)
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            n_classes = logits.shape[1]

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_targets.extend(labels.detach().cpu().numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())

    average_loss = total_loss / max(total_samples, 1)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    labels = list(range(n_classes)) if n_classes is not None else None
    per_class_f1 = f1_score(all_targets, all_preds, labels=labels, average=None, zero_division=0)
    return average_loss, macro_f1, per_class_f1


def compute_class_weights(labels, n_classes):
    labels = np.asarray(labels).squeeze().astype(np.int64)
    counts = np.bincount(labels, minlength=n_classes)
    total = counts.sum()
    weights = total / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)
