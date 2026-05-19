import numpy as np
import torch

from src.evaluate import predict_dataset
from src.models import build_model


def load_model_from_checkpoint(ckpt_path, model_name, n_classes, device):
    model = build_model(model_name, n_classes)
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state_dict = checkpoint

    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def softmax_ensemble(checkpoint_paths, model_name, n_classes, test_loader, device, weights=None):
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")

    if weights is None:
        weights = np.ones(len(checkpoint_paths), dtype=np.float64) / len(checkpoint_paths)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    assert len(weights) == len(checkpoint_paths), "weights length must match checkpoint_paths"
    assert np.isclose(weights.sum(), 1.0), "weights must sum to 1"

    averaged_probs = None
    for ckpt_path, weight in zip(checkpoint_paths, weights):
        model = load_model_from_checkpoint(ckpt_path, model_name, n_classes, device)
        _, probs = predict_dataset(model, test_loader, device)
        weighted_probs = weight * probs
        averaged_probs = weighted_probs if averaged_probs is None else averaged_probs + weighted_probs

    return averaged_probs.argmax(axis=1)
