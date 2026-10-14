# Extreme Stock Move Prediction

Binary classifier for extreme daily moves in Apple (AAPL) stock, defined as days where the absolute daily return exceeds 2%. Daily prices are downloaded from Yahoo Finance (2015–2024), validated against a schema, cleaned, labelled and split chronologically, then used to train models.

## Local DEV

**Show all commands:**
```bash
make help
```

**Build image, run tests and run the pipeline (retrieval, preprocessing, feature engineering, data split, random forest, temporal CNN, improvement, evaluation):**
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

## CI/CD

GitHub Actions ([`.github/workflows/cicd.yml`](.github/workflows/cicd.yml)) runs eleven jobs; each runs only if the jobs it depends on succeeded, and the three training jobs run in parallel:

```
ci > cd > integration > retrieve > preprocess > features > split ─┬─> random-forest ─┐
                                                                 ├─> temporal-cnn  ──┼─> evaluation
                                                                 └─> improvement   ──┘
```

1. **ci** — builds the image and runs the unit tests. Runs on every push and pull request.
2. **cd** — publishes the image to GitHub Container Registry as `ghcr.io/andmanousakis/stock-extreme:<environment>` and `:<commit>`. Runs on pushes only.
3. **integration** — pulls the published image and runs the integration tests against Yahoo Finance.
4. **retrieve** — runs data retrieval in the published image and keeps the raw data and log as a `data-<environment>-<date>-<commit>-r<run>.<attempt>` release, so every run and re-run gets its own release.
5. **preprocess** — downloads the raw data from that release, cleans it in the published image and adds the clean data and log to the same release.
6. **features** — downloads the clean data from that release, builds the features and next-day target in the published image and adds the feature dataset and log to the same release.
7. **split** — downloads the feature dataset from that release, splits it in time order (70% train, 15% val, 15% test) in the published image and adds the three sets and log to the same release.
8. **random-forest** — downloads the three sets from that release, trains a random forest on 10-day windows, tuned on the validation set, in the published image and adds the trained model and log to the same release.
9. **temporal-cnn** — downloads the three sets from that release, trains the temporal CNN on scaled 10-day sequences with early stopping on the validation set, in the published image and adds the trained model and log to the same release.
10. **improvement** — downloads the three sets from that release, trains temporal CNN variants (class weights, a tuned decision threshold, volatility features, a smaller regularised network), keeps the best by validation F1, and adds it with the comparison table and log to the same release.
11. **evaluation** — downloads the three sets and the three trained models from that release, scores them and three simple baselines (never extreme, always extreme, persistence) on the test set, shows the results on the run page, and adds the report and log to the same release. This is the only job that scores on the test set.

GitHub Actions builds from [`docker/deploy/Dockerfile`](docker/deploy/Dockerfile). [`docker/local/`](docker/local/) and the makefile are for local development only.

The environment comes from the branch:

| Branch | Environment |
|---|---|
| `prod` | prod |
| `uat` | uat |
| any other branch | dev |

The environment is passed to the container as `APP_ENV`, which selects the configs in `configs/<environment>/`. Dev retrieves a shorter date range with fewer retries and debug logging; uat and prod use the full range. Local runs through `make` use dev.