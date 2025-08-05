#!/bin/bash

docker run --rm -it \
    -v "$(pwd)":/shared \
    -v "$HOME/luxonis_ml":/root/luxonis_ml \
    luxonis-train-migrate-weights "$@"
