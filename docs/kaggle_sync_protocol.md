# Kaggle Synchronization Protocol

This document defines the VS Code to Kaggle synchronization workflow for ACCV Tensor Reloaded / Multi-Task MedMNIST experiments.

## Local To Kaggle Before Each Training Run

1. Confirm local changes are ready:
   ```bash
   git status
   ```

2. Push the current repository state:
   ```bash
   git push origin main
   ```

3. In Kaggle, run K-02 in `notebooks/kaggle_execution_template.ipynb` to clone or pull the ACCV repository.

4. Edit K-00 run config for the experiment:
   - `RUN_NAME`
   - `DATASETS_TO_RUN`
   - `MODEL_NAME`
   - `IMAGE_SIZE`
   - `PHASE1_EPOCHS`
   - `PHASE2_EPOCHS`
   - learning rates, loss, augmentation, scheduler, and TTA settings

5. Never manually upload data or checkpoints into Git.

6. Never manually upload data or checkpoints as replacement source files for the repository. Kaggle data should come from the attached Kaggle dataset, and checkpoints should be generated in Kaggle outputs.

## Kaggle To Local After Each Training Run

Download required run artifacts from Kaggle Outputs:

- `submission_{run_name}_{ts}.csv` to `submissions/`
- `results_{run_name}_{ts}.csv` and append its rows to `experiments/results_tracker.csv`
- `run_config_{run_name}_{ts}.json` to `experiments/`

Optional downloads:

- `{dataset}_test_probs_best.npy` only for ensemble or TTA analysis

Never commit:

- `*.pth`
- `checkpoints/`
- `data/`
- `*.npz`
- large `*.npy`

## Kaggle Checkpoint Traceability

Every Kaggle run must be recorded in `experiments/submission_log.csv` with this schema:

```csv
version,timestamp,kaggle_run_url,public_score,key_changes,notes
```

The `kaggle_run_url` is the permanent pointer back to the Kaggle run output. Use it later to find old checkpoints, logs, probability arrays, and exported configs without committing large artifacts to Git.

## Git Commit Examples After Each Run

Baseline:

```bash
git add experiments/results_tracker.csv experiments/submission_log.csv experiments/run_config_baseline.json submissions/submission_baseline.csv
git commit -m "Add baseline Kaggle run results"
```

EfficientNet:

```bash
git add experiments/results_tracker.csv experiments/submission_log.csv experiments/run_config_efficientnet.json submissions/submission_efficientnet.csv
git commit -m "Add EfficientNet Kaggle run results"
```

Loss experiments:

```bash
git add experiments/results_tracker.csv experiments/submission_log.csv experiments/run_config_loss_exp.json submissions/submission_loss_exp.csv
git commit -m "Add loss experiment Kaggle results"
```

Hyperparameter tuning:

```bash
git add experiments/results_tracker.csv experiments/submission_log.csv experiments/run_config_hparam.json submissions/submission_hparam.csv
git commit -m "Add hyperparameter tuning results"
```

Ensemble/TTA:

```bash
git add experiments/results_tracker.csv experiments/submission_log.csv experiments/run_config_ensemble_tta.json submissions/submission_ensemble_tta.csv
git commit -m "Add ensemble TTA submission results"
```

Ablation:

```bash
git add experiments/results_tracker.csv experiments/submission_log.csv ablation/figures experiments/run_config_ablation.json
git commit -m "Add ablation study results"
```

## Quick Checklist

Before pushing:

- [ ] `git status` has no `*.pth` or `*.npz`
- [ ] `.gitignore` covers `checkpoints/` and `data/`
- [ ] `experiments/results_tracker.csv` updated

After Kaggle run:

- [ ] submission CSV downloaded
- [ ] results rows appended
- [ ] run config JSON saved
- [ ] Kaggle URL saved
- [ ] public leaderboard score recorded
