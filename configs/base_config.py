import os

import torch


IS_KAGGLE = os.path.exists("/kaggle")

BASE_CONFIG = {
    "seed": 42,
    "data_root": "/kaggle/input/tensor-reloaded-multi-task-med-mnist/data/" if IS_KAGGLE else "data/",
    "output_dir": "/kaggle/working/checkpoints/" if IS_KAGGLE else "checkpoints/",
    "results_csv": "experiments/results_tracker.csv",
    "submission_dir": "/kaggle/working/submissions/" if IS_KAGGLE else "submissions/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "is_kaggle": IS_KAGGLE,
}
