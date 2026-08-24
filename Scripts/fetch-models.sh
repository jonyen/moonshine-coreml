#!/bin/bash
# Downloads a release's compiled models: Scripts/fetch-models.sh 0.1.0 [DEST]
set -euo pipefail
VERSION="${1:?version, e.g. 0.1.0}"
DEST="${2:-$(cd "$(dirname "$0")/.." && pwd)/build}"
URL="https://github.com/jonyen/moonshine-coreml/releases/download/v$VERSION/moonshine-tiny-coreml-v$VERSION.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL "$URL" -o "$TMP/models.zip"
mkdir -p "$DEST"
unzip -oq "$TMP/models.zip" -d "$DEST"
echo "models in $DEST:"; ls "$DEST"
