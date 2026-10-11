COMPOSE := docker compose -f docker/local/docker-compose.yml

export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)

.DEFAULT_GOAL := help

.PHONY: help stock-extreme-dev-build-up stock-extreme-dev-up stock-extreme-dev-down stock-extreme-dev-clean-docker stock-extreme-dev-wipe-docker stock-extreme-dev-logs

# Show available commands and their descriptions.
help:
	@echo "Usage: make <command>"
	@echo ""
	@awk '/^# / { desc = substr($$0, 3); next } /^[a-zA-Z0-9_-]+:/ && desc { split($$1, target, ":"); printf "  \033[36m%-32s\033[0m %s\n", target[1], desc } { desc = "" }' $(MAKEFILE_LIST)

# Build the stock-extreme:dev image and start the container.
stock-extreme-dev-build-up:
	$(COMPOSE) up -d --build

# Start the container from the existing image.
stock-extreme-dev-up:
	$(COMPOSE) up -d

# Stop and remove the container.
stock-extreme-dev-down:
	$(COMPOSE) down

# Remove this project's container, the stock-extreme:dev image and leftover containers. Keeps volumes.
stock-extreme-dev-clean-docker:
	$(COMPOSE) down --rmi all --remove-orphans

# Same as clean-docker, plus delete this project's Docker volumes.
stock-extreme-dev-wipe-docker:
	$(COMPOSE) down --rmi all --remove-orphans --volumes

# Follow the container logs.
stock-extreme-dev-logs:
	$(COMPOSE) logs -f
