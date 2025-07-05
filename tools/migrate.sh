#!/bin/bash

set -e

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [<CONFIG_OVERRIDES>...]

Options:
  --weights <file>              Path to the checkpoint to migrate
  --config <file>               Path to config file (optional)
  --out-dir <file>              Path to output weights file (optional, defaults to 'new_{weights}')
  --old-execution-order <file>  Path to old execution order file (optional)
  --new-execution-order <file>  Path to new execution order file (optional)
  --help                        Display this help message and exit

Any remaining arguments will be treated as config overrides.
EOF
}

WEIGHTS=""
CONFIG=""
OUT_DIR=""
OLD_EXECUTION_ORDER=""
NEW_EXECUTION_ORDER=""
OPTS=""


# Argument parsing
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --weights)
            WEIGHTS="$WEIGHTS $2"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --old-execution-order)
            OLD_EXECUTION_ORDER="$2"
            shift 2
            ;;
        --new-execution-order)
            NEW_EXECUTION_ORDER="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            if [[ "$1" == --* ]]; then
                echo "Error: Unknown option '$1'."
                show_help
                exit 1
            fi
            OPTS+="$1 "
            shift
            ;;
    esac
done

if [[ -z $WEIGHTS ]]; then
    echo "Error: --weights must be specified."
    exit 1
fi

if [[ ! -f $CONFIG ]]; then
    echo "Error: Config file '$CONFIG' does not exist."
    exit 1
fi

for WEIGHT in $WEIGHTS; do
    if [[ ! -f $WEIGHT ]]; then
        echo "Error: Weights file '$WEIGHT' does not exist."
        exit 1
    fi
done

if [[ -z $OUT_DIR ]]; then
    OUT_DIR="migrated_weights"
fi

mkdir -p "${OUT_DIR}"

OPTS=$(echo "$OPTS" | xargs)

# source ~/miniconda3/bin/activate

. old/bin/activate
# conda activate luxonis-train-prod

if [[ -z $OLD_EXECUTION_ORDER ]]; then
    python tools/generate_execution_order.py --config "$CONFIG" \
        --output old_execution_order.txt $OPTS > /dev/null

    echo "Old execution order generated: old_execution_order.txt"
fi

. new/bin/activate
# conda activate luxonis-train

if [[ -z $NEW_EXECUTION_ORDER ]]; then
    python tools/generate_execution_order.py --config "$CONFIG" \
        --output new_execution_order.txt \
        model.predefined_model.params.weights none $OPTS > /dev/null

    echo "New execution order generated: new_execution_order.txt"
fi


for WEIGHT in $WEIGHTS; do
    echo "Migrating $WEIGHT"

    python tools/migrate_weights.py \
        --old-execution-order old_execution_order.txt \
        --new-execution-order new_execution_order.txt \
        --weights "$WEIGHT" \
        --out-dir "$OUT_DIR"
done

echo "Weights migrated to: $OUT_DIR/"
