.PHONY: all lint generate-mocks generate-protos test

all: format lint test

requirements:
	python -m venv venv
	. venv/bin/activate
	pip install -r requirements.txt

	@if [[ $(shell uname -s) == "Darwin" && $(shell uname -m) == "arm64" ]]; then \
		echo "Detected macOS ARM (Apple Silicon). Installing TensorFlow for ARM."; \
		pip install -r requirements_tf_arm.txt; \
	else \
		echo "Detected non-ARM system. Installing standard TensorFlow."; \
		pip install -r requirements_tf.txt; \
	fi

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

build-pkg:
	@echo "Building distribution"
	python3 -m build

publish:
	@echo "Publishing to pip"
	python3 -m twine upload --repository pypi dist/*