#!/usr/bin/env bash
# Скачивание Kaggle "Huge Stock Market Dataset" в data/raw/kaggle/.
#
# Pre-requisites:
#   1. pip install kaggle  (уже в pyproject.toml dependencies)
#   2. Kaggle API token:
#        - https://www.kaggle.com/settings  ->  Create New API Token
#        - сохранить kaggle.json в ~/.kaggle/kaggle.json
#        - chmod 600 ~/.kaggle/kaggle.json
#   3. Принять Terms на странице датасета:
#        https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
#
# Размер: ~700 MB unzipped, ~7000 файлов в Stocks/ и ~1300 в ETFs/.

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
