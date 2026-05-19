SMALL_DATASETS = {"retinamnist", "breastmnist"}
NO_VFLIP_DATASETS = {"organamnist", "organcmnist", "organsmnist", "octmnist"}
FOCUS_TUNING_DATASETS = {"dermamnist", "retinamnist", "breastmnist", "octmnist"}

DATASET_CONFIGS = {
    "pathmnist": {
        "n_classes": 9,
        "task": "multiclass",
        "is_rgb": True,
        "n_train": 89996,
        "imbalance_ratio": 1.63,
    },
    "dermamnist": {
        "n_classes": 7,
        "task": "multiclass",
        "is_rgb": True,
        "n_train": 7007,
        "imbalance_ratio": 58.66,
    },
    "octmnist": {
        "n_classes": 4,
        "task": "multiclass",
        "is_rgb": False,
        "n_train": 97477,
        "imbalance_ratio": 5.93,
    },
    "pneumoniamnist": {
        "n_classes": 2,
        "task": "binary",
        "is_rgb": False,
        "n_train": 4708,
        "imbalance_ratio": 2.87,
    },
    "retinamnist": {
        "n_classes": 5,
        "task": "ordinal",
        "is_rgb": True,
        "n_train": 1080,
        "imbalance_ratio": 7.36,
    },
    "breastmnist": {
        "n_classes": 2,
        "task": "binary",
        "is_rgb": False,
        "n_train": 546,
        "imbalance_ratio": 2.71,
    },
    "bloodmnist": {
        "n_classes": 8,
        "task": "multiclass",
        "is_rgb": True,
        "n_train": 11959,
        "imbalance_ratio": 2.74,
    },
    "tissuemnist": {
        "n_classes": 8,
        "task": "multiclass",
        "is_rgb": False,
        "n_train": 165466,
        "imbalance_ratio": 9.04,
    },
    "organamnist": {
        "n_classes": 11,
        "task": "multiclass",
        "is_rgb": False,
        "n_train": 34581,
        "imbalance_ratio": 4.54,
    },
    "organcmnist": {
        "n_classes": 11,
        "task": "multiclass",
        "is_rgb": False,
        "n_train": 13000,
        "imbalance_ratio": 5.01,
    },
    "organsmnist": {
        "n_classes": 11,
        "task": "multiclass",
        "is_rgb": False,
        "n_train": 13940,
        "imbalance_ratio": 5.64,
    },
}
