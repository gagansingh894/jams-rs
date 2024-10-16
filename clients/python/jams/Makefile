.PHONY: all lint generate-mocks generate-protos test

all: format lint test

mypy:
	mypy .

ruff:
	ruff check --fix

format:
	@echo "Formatting code..."
	ruff format

lint: mypy ruff

test:
	@echo "Running tests..."
	pytest tests
