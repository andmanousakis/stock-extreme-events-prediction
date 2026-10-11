# Extreme Stock Move Prediction

Binary classifier for extreme daily moves in Apple (AAPL) stock, defined as days where the absolute daily return exceeds 2%. Daily prices are downloaded from Yahoo Finance (2015–2024), validated against a schema, cleaned, labelled and split chronologically, then used to train models.

## How to Run

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

Starting the container runs the unit tests, then data retrieval only if they all pass. On GitHub, CI runs the unit tests on every push; pushes to `dev` then publish the image (CD) and run the integration tests, which call Yahoo Finance, against it. Raw data is saved to `data/raw/apple_stock_raw.xlsx` and logs to `logs/pipeline.log`.
