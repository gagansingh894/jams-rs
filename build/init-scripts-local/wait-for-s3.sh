#!/usr/bin/env bash
# wait-for-it.sh

set -e

host="$1"
shift
cmd="$@"

until awslocal s3 ls; do
  >&2 echo "S3 is unavailable - sleeping"
  sleep 5
done

>&2 echo "S3 is up - executing command"
exec $cmd