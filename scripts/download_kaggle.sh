#!/usr/bin/env bash

set -euo pipefail

DEST="data/raw/kaggle"
DATASET="borismarjanovic/price-volume-data-for-all-us-stocks-etfs"

mkdir -p "$DEST"

if [ -d "$DEST/Stocks" ] && [ "$(ls -A "$DEST/Stocks" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "Stocks/ already populated ($(ls "$DEST/Stocks" | wc -l) files). Skipping download."
    exit 0
fi

echo "Downloading Kaggle dataset: $DATASET (~700 MB) ..."
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip

echo "Done."
echo "  Stocks: $(ls "$DEST/Stocks" 2>/dev/null | wc -l) files"
echo "  ETFs:   $(ls "$DEST/ETFs" 2>/dev/null | wc -l) files"
