# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-17

First release promoted to `uat`.

### Added
- Model classes in `src/models/`: `TemporalCNNModel` with the network chosen from config (`src/models/networks/`).
- Training and evaluation step: scores every model on train, val and test with F1, precision, recall, MCC, ROC AUC and accuracy, and writes metrics and predictions tables tagged with `run_id` and `run_date`.
- Model selection step: picks the best model of the latest run by validation MCC; test is reported only.
- MLflow tracking and model registry on DagsHub (`src/mlops/`): one run per model with settings, metrics, training curves, confusion matrices, ROC and precision-recall curves; best model registered as `stock-extreme-model` with an alias per environment.
- Parquet table storage on DagsHub (`utils/storage_io.py`), one file per table and run, so history is kept.
- Class weights, weight decay and validation-tuned decision threshold for the temporal CNN.
- Volatility, intraday range, overnight gap and volume-ratio features.
- Modelling and deployment docs in `docs/`, and this changelog.

### Changed
- Pipeline steps moved to `src/pipeline/`.
- Daily returns, extreme-event labels and the next-day target are now built in data preprocessing.
- Threshold tuning, model selection and registry promotion use validation MCC instead of F1.
- Date range extended to 1990 onwards (2005 onwards in dev), up to the latest trading day.
- CI/CD runs training-evaluation and model-selection as separate jobs; releases hold logs and models only.
- CI/CD runs manually (`workflow_dispatch`) instead of on every push and pull request.

### Removed
- Random forest and baseline temporal CNN.
- Excel files as pipeline inputs and outputs.
- Separate improvement and evaluation scripts, replaced by config-driven models and the model selection step.

## [0.2.0] - 2026-10-14

### Added
- Data preprocessing: schema validation, sorting, de-duplication and forward-filling on trading days only.
- Feature engineering: daily returns, extreme events (absolute daily return above 2%) and next-day target.
- Time-ordered 70/15/15 train, validation and test split.
- Random forest and temporal CNN on 10-day windows, with improvement experiments and test-set evaluation against simple baselines.
- Unit tests for leakage and time order, and one end-to-end integration test.

## [0.1.0] - 2026-10-11

### Added
- Data retrieval from Yahoo Finance with retries, exponential backoff and schema validation.
- Per-environment configs (`dev`, `uat`, `prod`) selected by `APP_ENV`.
- Docker setup for local development and deployment.
- GitHub Actions CI/CD: unit tests, image publishing to GitHub Container Registry, integration tests and per-run data releases.

[Unreleased]: https://github.com/andmanousakis/stock-extreme-events-prediction/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/andmanousakis/stock-extreme-events-prediction/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/andmanousakis/stock-extreme-events-prediction/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/andmanousakis/stock-extreme-events-prediction/releases/tag/v0.1.0
