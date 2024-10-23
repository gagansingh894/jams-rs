#!/bin/bash

set -e

echo "Waiting for LocalStack services to be available..."
while ! awslocal s3 ls > /dev/null 2>&1; do
    sleep 5
done

echo "Initializing LocalStack..."
awslocal s3 mb s3://jamsmodelstore