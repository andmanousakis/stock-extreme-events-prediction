# Extreme Stock Move Prediction

Binary classifier for extreme daily moves in Apple (AAPL) stock, defined as days where the absolute daily return exceeds 2%. Daily prices are downloaded from Yahoo Finance (2015–2024), validated against a schema, cleaned, labelled and split chronologically, then used to train models.

## Local DEV

**Show all commands:**
```bash
make help
```

**Build image, run tests and start data retrieval:**
```bash
make stock-extreme-dev-build-up
```

**Run tests and start data retrieval (image already built):**
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

GitHub Actions ([`.github/workflows/cicd.yml`](.github/workflows/cicd.yml)) runs four jobs in sequence; each runs only if the previous one succeeded:

```
ci  >  cd  >  integration  >  retrieve
```

1. **ci** — builds the image and runs the unit tests. Runs on every push and pull request.
2. **cd** — publishes the image to GitHub Container Registry as `ghcr.io/andmanousakis/stock-extreme:<environment>` and `:<commit>`. Runs on pushes only.
3. **integration** — pulls the published image and runs the integration tests against Yahoo Finance.
4. **retrieve** — runs data retrieval in the published image and keeps the raw data and log as a `data-<environment>-<date>-<commit>` release.

GitHub Actions builds from [`docker/deploy/Dockerfile`](docker/deploy/Dockerfile). [`docker/local/`](docker/local/) and the makefile are for local development only.

The environment comes from the branch:

| Branch | Environment |
|---|---|
| `prod` | prod |
| `uat` | uat |
| any other branch | dev |

The environment is passed to the container as `APP_ENV`, which selects the configs in `configs/<environment>/`. Dev retrieves a shorter date range with fewer retries and debug logging; uat and prod use the full range. Local runs through `make` use dev.