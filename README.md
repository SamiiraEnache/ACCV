# ACCV — Tensor Reloaded MedMNIST

## Overview

Repository scaffold for the Kaggle Tensor Reloaded / Multi-Task MedMNIST Classification workflow.

## Workflow

This repository follows the Jira task sequence. TASK-001 creates the project structure, configuration files, experiment trackers, and repository hygiene files only.

## Local Development

Use the local `data/`, `checkpoints/`, and `submissions/` paths when running outside Kaggle. Data and checkpoint artifacts are intentionally ignored by Git.

## Kaggle Training

Kaggle runs should use `/kaggle/input/tensor-reloaded-multi-task-med-mnist/data/` for input data and `/kaggle/working/` for generated checkpoints and submissions.

## Kaggle Synchronization Protocol

See `docs/kaggle_sync_protocol.md`.

## Git Rules

Do not commit datasets, checkpoints, model weights, NumPy artifacts, environment files, or notebook checkpoints.

## Reproducibility

The base configuration defines seed `42`, environment-aware paths, and automatic device selection.

## Team

Add team members and responsibilities here as the workflow progresses.
