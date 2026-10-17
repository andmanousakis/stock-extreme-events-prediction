# Extreme Stock Move Prediction

Binary classifier for extreme daily moves in Apple (AAPL) stock, defined as days where the absolute daily return exceeds 2%. Daily prices are downloaded from Yahoo Finance (from 1990 to the latest trading day; 2005 onwards in dev), validated against a schema, cleaned, labelled and split chronologically, then used to train models.

## Documentation

- [Modelling](docs/modelling.md): data, features, split, model, metrics and selection.
- [Deployment](docs/deployment.md): environments, CI/CD, Docker, storage, MLflow and secrets.
- [Changelog](CHANGELOG.md)

## Local DEV

**Show all commands:**
```bash
make help
```

**Build image, run tests and run the pipeline (retrieval, preprocessing, feature engineering, data split, training and evaluation, model selection, model monitoring):**
```bash
make stock-extreme-dev-build-up
```

**Run tests and run the pipeline (image already built):**
```bash
make stock-extreme-dev-up
```

**Follow the logs:**
```bash
make stock-extreme-dev-logs
```

**Stop and remove the container:**
```bash
make stock-extreme-dev-down
```

**Remove container and image:**
```bash
make stock-extreme-dev-clean-docker
```

**Remove container, image and volumes:**

> [!WARNING]
> Permanently deletes this project's Docker volumes and any data stored in them. Files in the project folder (`data/`, `logs/`) are not affected.

```bash
make stock-extreme-dev-wipe-docker
```

## Deployment - CI/CD

GitHub Actions ([`.github/workflows/cicd.yml`](.github/workflows/cicd.yml)) runs manually: open **Actions → CI/CD → Run workflow** and pick the branch. It runs ten jobs in sequence; each runs only if the previous one succeeded:

```
ci > cd > integration > retrieve > preprocess > features > split > training-evaluation > model-selection > model-monitoring
```

1. **ci** — builds the image and runs the unit tests.
2. **cd** — publishes the image to GitHub Container Registry as `ghcr.io/andmanousakis/stock-extreme:<environment>` and `:<commit>`.
3. **integration** — pulls the published image and runs the integration tests against Yahoo Finance.
4. **retrieve** — downloads AAPL prices, checks them against the schema, writes the raw Parquet table on DagsHub storage, and creates a `data-<environment>-<date>-<commit>-r<run>.<attempt>` release for the run's logs and models.
5. **preprocess** — reads the raw table, cleans it, calculates daily returns, labels next-day extreme events, and writes the clean table.
6. **features** — reads the clean table, adds volatility, range, gap and volume-ratio features, and writes the feature table.
7. **split** — reads the feature table and splits it in time order (70% train, 15% val, 15% test) into three tables.
8. **training-evaluation** — reads the three tables, trains every model in the config, scores it on train, val and test (F1, precision, recall, MCC, ROC AUC, accuracy), logs it to MLflow, writes its metrics and predictions tables (tagged with the run ID and date), shows the test results on the run page, and adds the models to the release.
9. **model-selection** — reads this run's metrics from the metrics table, picks the best model by validation MCC, registers it in MLflow as a new version, and adds it to the same release.
10. **model-monitoring** — scores the deployed model and the newly registered one on this run's val and test tables, moves the environment's alias to the new model if its validation MCC is higher, writes the comparison to the monitoring table, and shows it on the run page.

GitHub Actions builds from [`docker/deploy/Dockerfile`](docker/deploy/Dockerfile). [`docker/local/`](docker/local/) and the makefile are for local development only.

The environment comes from the branch:

| Branch | Environment |
|---|---|
| `prod` | prod |
| `uat` | uat |
| any other branch | dev |

The environment is passed to the container as `APP_ENV`, which selects the configs in `configs/<environment>/`. Dev retrieves a shorter date range with fewer retries and debug logging; uat and prod use the full range. Local runs through `make` use dev.