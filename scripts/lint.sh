#!/usr/bin/env bash
## Format all code directories in the repostitory using cargo fmt.

for DIR in */; do
    DIRNAME=$(basename "$DIR")
    echo "==> $DIRNAME <=="
    (cd $DIR && cargo clippy --all-targets -- -D errors) # todo: change to warnings
done

echo "Format complete."