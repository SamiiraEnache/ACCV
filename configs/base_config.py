import os
from pathlib import Path

import torch


IS_KAGGLE = os.path.exists("/kaggle/input") and os.path.exists("/kaggle/working")

if IS_KAGGLE:
    candidate_paths = [
        "/kaggle/input/competitions/tensor-reloaded-multi-task-med-mnist/data/",
        "/kaggle/input/tensor-reloaded-multi-task-med-mnist/data/",
    ]

    DATA_ROOT = None
    for path in candidate_paths:
        if (Path(path) / "pathmnist.npz").exists():
            DATA_ROOT = path
            break

    if DATA_ROOT is None:
        DATA_ROOT = candidate_paths[0]
else:
    DATA_ROOT = "data/"

BASE_CONFIG = {
    "seed": 42,
    "data_root": DATA_ROOT,
    "output_dir": "/kaggle/working/checkpoints/" if IS_KAGGLE else "checkpoints/",
    "results_csv": "experiments/results_tracker.csv",
    "submission_dir": "/kaggle/working/submissions/" if IS_KAGGLE else "submissions/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "is_kaggle": IS_KAGGLE,
}
