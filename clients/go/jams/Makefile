.PHONY: all lint generate-mocks generate-protos test

all: lint generate-mocks generate-protos test

lint:
	#todo: fix
	@echo "Running linter..."
	#golangci-lint run

generate-mocks:
	@echo "Generating mocks..."
	go generate ./...

generate-protos:
	@echo "Generating protobufs..."
	protoc --go_out=pkg/pb --go-grpc_out=pkg/pb -I . grpc/*.proto

test:
	@echo "Running tests..."
	go test ./...
