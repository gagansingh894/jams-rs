#!/bin/bash

# Wait for Azurite to start
echo "Waiting for Azurite to start..."
sleep 10  # Give Azurite some time to initialize

# Create a storage account and a container
echo "Creating Azure Storage account and container..."

# Create a container named 'my-container'
az storage container create --name jamsmodelstore --account-name devstoreaccount1 --account-key Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw== --blob-endpoint http://localhost:10000


#az storage container create -n test --connection-string "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;"
# Optionally, create more containers or perform additional setups
# az storage container create --name another-container --account-name test --account-key test --endpoint http://localhost:10000

echo "Azure Storage setup complete."