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
	cargo clippy --all-targets --all-features -- -D warnings

nextest:
	@echo "Testing all projects with cargo nextest"
	cargo nextest run

test:
	@echo "Testing all projects with cargo test"
	cargo test

check-gpu-linux:
	sudo lshw -C display

all: format lint test

generate-proto:
	@echo "Generating proto files for Go"
	protoc -I=internal/jams-proto/proto/api/v1/ --go_out=clients/go/jams --go-grpc_out=clients/go/jams jams.proto
	@echo "Generating proto files for Python"
	python -m grpc_tools.protoc -I=internal/jams-proto/proto/api/v1/ --python_out=clients/python/jams/proto --pyi_out=clients/python/jams/proto --grpc_python_out=clients/python/jams/proto jams.proto
#	@echo "Generating proto files for TypeScript"
#	protoc -I=jams-serve/proto/api/v1/ --plugin="$(which protoc-gen-ts)" --ts_proto_opt=esModuleInterop=true --ts_proto_out="clients/typescript/generated" jams.proto
	@echo "Generating proto files for Java"
	protoc -I=internal/jams-proto/proto/api/v1 --java_out=clients/java jams.proto
