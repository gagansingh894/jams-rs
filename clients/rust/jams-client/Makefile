install:
	# Install if needed
	#@echo "Updating rust toolchain"
	#rustup update stable
	#rustup default stable

rust-version:
	@echo "Rust command-line utility versions:"
	rustc --version 			#rust compiler
	cargo --version 			#rust package manager
	rustfmt --version			#rust code formatter
	rustup --version			#rust toolchain manager
	clippy-driver --version		#rust linter

format:
	@echo "Formatting all projects with cargo"
	cargo fmt --

lint:
	@echo "Linting all projects with cargo"
	@rustup component add clippy 2> /dev/null
	cargo clippy --package jams-client --all-targets --all-features -- -D warnings

nextest:
	@echo "Testing all projects with cargo nextest"
	cargo nextest run -p jams-client --retries 2

test:
	@echo "Testing all projects with cargo test"
	cargo test

all: format lint test
