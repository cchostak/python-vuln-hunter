#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-data/raw/kaggle_vuln}
DATASET_SLUG="jiscecseaiml/vulnerability-fix-dataset"
ZIP_PATH="/tmp/vulnerability-fix-dataset.zip"

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "KAGGLE_USERNAME and KAGGLE_KEY must be set (from your Kaggle API token)." >&2
  exit 1
fi

echo "Downloading ${DATASET_SLUG} to ${ZIP_PATH}..."
curl -fL -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
  "https://www.kaggle.com/api/v1/datasets/download/${DATASET_SLUG}" \
  -o "${ZIP_PATH}"

echo "Extracting to ${DATA_DIR}..."
mkdir -p "${DATA_DIR}"
unzip -o "${ZIP_PATH}" -d "${DATA_DIR}" >/dev/null
rm -f "${ZIP_PATH}"
echo "Done. Files available under ${DATA_DIR}"
