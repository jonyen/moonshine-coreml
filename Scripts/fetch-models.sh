#!/bin/bash
# Downloads a release's compiled models: Scripts/fetch-models.sh VERSION [tiny|base] [DEST]
# The model defaults to tiny; DEST defaults to build/ for tiny, build-<model>/ otherwise.
set -euo pipefail
VERSION="${1:?version, e.g. 0.2.0}"
shift
MODEL="tiny"
if [ "${1:-}" = "tiny" ] || [ "${1:-}" = "base" ]; then MODEL="$1"; shift; fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_DEST="$ROOT/build"; [ "$MODEL" != tiny ] && DEFAULT_DEST="$ROOT/build-$MODEL"
DEST="${1:-$DEFAULT_DEST}"
URL="https://github.com/jonyen/moonshine-coreml/releases/download/v$VERSION/moonshine-$MODEL-coreml-v$VERSION.zip"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
curl -fL "$URL" -o "$TMP/models.zip"
mkdir -p "$DEST"
unzip -oq "$TMP/models.zip" -d "$DEST"
echo "models in $DEST:"; ls "$DEST"
