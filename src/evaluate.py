import numpy as np
import torch
from scipy.stats import hmean
from sklearn.metrics import f1_score


def _batch_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def predict_dataset(model, loader, device):
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for batch in loader:
            images = _batch_images(batch).to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    return np.concatenate(predictions), np.concatenate(probabilities)


def compute_harmonic_mean(f1_dict):
    values = np.asarray(list(f1_dict.values()), dtype=np.float64)
    if values.size == 0 or np.any(values <= 0):
        return 0.0
    return float(hmean(values))


def evaluate_all_datasets(models_dict, loaders_dict, labels_dict, device):
    predictions_dict = {}
    probabilities_dict = {}
    f1_dict = {}

    for dataset_name, model in models_dict.items():
        preds, probs = predict_dataset(model, loaders_dict[dataset_name], device)
        labels = np.asarray(labels_dict[dataset_name]).squeeze().astype(np.int64)
        predictions_dict[dataset_name] = preds
        probabilities_dict[dataset_name] = probs
        f1_dict[dataset_name] = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "predictions": predictions_dict,
        "probabilities": probabilities_dict,
        "f1": f1_dict,
        "harmonic_mean": compute_harmonic_mean(f1_dict),
    }
