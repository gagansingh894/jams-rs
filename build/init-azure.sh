#!/bin/bash

# Wait for Azurite to start
echo "Waiting for Azurite to start..."
sleep 10  # Give Azurite some time to initialize

# Create a storage account and a container
echo "Creating Azure Storage account and container..."

# Create a container named 'my-container'
az storage container create --name model-store --account-name test --account-key test --endpoint http://localhost:10000

# Optionally, create more containers or perform additional setups
# az storage container create --name another-container --account-name test --account-key test --endpoint http://localhost:10000

echo "Azure Storage setup complete."