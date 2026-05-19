from pathlib import Path

import numpy as np
import pandas as pd

from configs.base_config import BASE_CONFIG
from configs.dataset_configs import DATASET_CONFIGS


def _find_npz_path(data_root, dataset_name):
    data_root = Path(data_root)
    candidates = [
        data_root / f"{dataset_name}.npz",
        data_root / dataset_name / f"{dataset_name}.npz",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find {dataset_name}.npz under {data_root}")


def _get_test_image_count(dataset_name):
    npz_path = _find_npz_path(BASE_CONFIG["data_root"], dataset_name)
    with np.load(npz_path) as data:
        if "test_images" not in data.files:
            raise KeyError(f"{npz_path} does not contain test_images")
        return len(data["test_images"])


def build_submission_csv(predictions_dict, output_path):
    expected_datasets = set(DATASET_CONFIGS.keys())
    provided_datasets = set(predictions_dict.keys())

    missing = expected_datasets - provided_datasets
    extra = provided_datasets - expected_datasets
    if missing or extra:
        raise ValueError(f"Dataset mismatch. Missing: {sorted(missing)}. Extra: {sorted(extra)}")

    rows = []
    expected_total_rows = 0
    global_id = 0

    for dataset_name in DATASET_CONFIGS.keys():
        predictions = np.asarray(predictions_dict[dataset_name]).squeeze().astype(np.int64)
        test_image_count = _get_test_image_count(dataset_name)
        expected_total_rows += test_image_count

        if len(predictions) != test_image_count:
            raise ValueError(
                f"{dataset_name} has {len(predictions)} predictions, expected {test_image_count}"
            )

        for id_image_in_task, label in enumerate(predictions):
            rows.append(
                {
                    "id": global_id,
                    "id_image_in_task": id_image_in_task,
                    "task_name": dataset_name,
                    "label": int(label),
                }
            )
            global_id += 1

    if len(rows) != expected_total_rows:
        raise ValueError(f"Submission has {len(rows)} rows, expected {expected_total_rows}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(rows, columns=["id", "id_image_in_task", "task_name", "label"])
    submission.to_csv(output_path, index=False)
    return submission
